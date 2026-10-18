import argparse
import csv
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mutual_info_score
from torch import nn


def incidence(x, metadata, bank):
    columns = []
    for edge in bank:
        if edge['kind'] == 'mechanism':
            membership = metadata[:, edge['column']].astype(str) == edge['value']
        else:
            projection = x[:, edge['features']]
            membership = np.linalg.norm(projection - np.array(edge['center']), axis=1) <= edge['radius']
        columns.append(membership)
    return np.array(columns, dtype=np.float32).T


def conditional_information(edge, labels, batch):
    return sum((batch == b).mean() * mutual_info_score(edge[batch == b], labels[batch == b]) for b in np.unique(batch))


def build_bank(x, metadata, y, batch, vx, vm, vy, vb, args, dependency_x):
    rng = np.random.default_rng(args.seed)
    correlation = np.atleast_2d(np.nan_to_num(np.corrcoef(dependency_x.T)))
    distance = np.clip(1 - np.abs(correlation), 0, 1)
    np.fill_diagonal(distance, 0)
    groups = fcluster(linkage(squareform(distance, checks=False), method='average'), .5, criterion='distance') if x.shape[1] > 1 else np.ones(1)
    bank = []
    for column in range(metadata.shape[1]):
        for value in np.unique(metadata[:, column].astype(str)):
            if np.mean(metadata[:, column].astype(str) == value) <= args.maximum_fraction:
                bank.append({'kind': 'mechanism', 'column': column, 'value': str(value), 'stability': 1.})
    batches = np.unique(batch)
    for group in np.unique(groups):
        features = np.flatnonzero(groups == group).tolist()
        projected = x[:, features]
        clusters = min(args.clusters, len(x))
        model = KMeans(clusters, n_init=10, random_state=args.seed).fit(projected)
        for center in model.cluster_centers_:
            distances = np.linalg.norm(projected - center, axis=1)
            radius = float(np.quantile(distances, min(args.maximum_fraction, 1 / clusters)))
            original = distances <= radius
            stability = []
            for _ in range(args.bootstrap):
                drawn = rng.choice(batches, len(batches), replace=True)
                indices = np.concatenate([np.flatnonzero(batch == b) for b in drawn])
                sample_model = KMeans(clusters, n_init=1, random_state=int(rng.integers(2 ** 31))).fit(projected[indices])
                updated = sample_model.cluster_centers_[np.linalg.norm(sample_model.cluster_centers_ - center, axis=1).argmin()]
                candidate = np.linalg.norm(projected - updated, axis=1) <= radius
                stability.append(float(np.logical_and(original, candidate).sum() / max(1, np.logical_or(original, candidate).sum())))
            bank.append({'kind': 'data', 'features': features, 'center': center.tolist(), 'radius': radius,
                         'stability': float(np.mean(stability))})
    validation = incidence(vx, vm, bank)
    for j, edge in enumerate(bank):
        observed = conditional_information(validation[:, j], vy, vb)
        null = []
        for _ in range(args.permutations):
            shuffled = vy.copy()
            for b in np.unique(vb):
                ids = np.flatnonzero(vb == b)
                shuffled[ids] = rng.permutation(shuffled[ids])
            null.append(conditional_information(validation[:, j], shuffled, vb))
        edge['information'] = float(observed - np.mean(null))
        edge['qualified'] = bool(edge['stability'] >= args.stability and edge['information'] > args.information)
        edge['weight'] = float(max(1e-4, edge['stability'] * max(0., edge['information'])))
    return bank


def relation(h, weights, query=None):
    degree = h @ weights
    if query is None:
        raw = (h * weights) @ h.T / (torch.sqrt(degree[:, None] * degree[None]) + 1e-6)
        rough = raw.clone()
        rough.fill_diagonal_(1.)
        return raw, rough
    qdegree = query @ weights
    return (query * weights) @ h.T / (torch.sqrt(qdegree[:, None] * degree[None]) + 1e-6)


def rough_bounds(s, membership):
    lower = torch.maximum(1 - s, membership[None]).min(-1).values
    upper = torch.minimum(s, membership[None]).max(-1).values
    return lower, upper


class Hypergraph(nn.Module):
    def __init__(self, features, weights, gates):
        super().__init__()
        self.input = nn.Linear(features, 32)
        self.layers = nn.ModuleList([nn.Linear(32, 32), nn.Linear(32, 32)])
        self.head = nn.Linear(32, 1)
        initial = torch.tensor(weights, dtype=torch.float32).clamp_min(1e-5)
        self.edge_parameter = nn.Parameter(torch.log(torch.expm1(initial)))
        self.register_buffer('gates', torch.tensor(gates, dtype=torch.float32))

    def forward(self, x, h, frequency, query_x=None, query_h=None):
        weight = nn.functional.softplus(self.edge_parameter) * self.gates
        hidden = self.input(x).relu()
        query = self.input(query_x).relu() if query_x is not None else None
        for layer in self.layers:
            edge = h.T @ (hidden * frequency[:, None]) / (h.T @ frequency).clamp_min(1e-6)[:, None]
            message = (h * weight) @ edge / (h @ weight).clamp_min(1e-6)[:, None]
            if query is not None:
                qmessage = (query_h * weight) @ edge / (query_h @ weight).clamp_min(1e-6)[:, None]
                query = (query + layer(qmessage)).relu()
            hidden = (hidden + layer(message)).relu()
        return self.head(hidden)[:, 0].sigmoid(), self.head(query)[:, 0].sigmoid() if query is not None else None, weight


def prepare(args):
    rows = list(csv.DictReader(open(args.source, encoding='utf-8')))
    features = [k for k in rows[0] if k not in ('label', 'batch', 'split') and not k.startswith('meta_')]
    mechanisms = [k for k in rows[0] if k.startswith('meta_')]
    x = np.array([[float(row[k]) if row[k] else np.nan for k in features] for row in rows])
    split = np.array([r['split'] for r in rows])
    train = split == 'train'
    median = np.nanmedian(x[train], axis=0)
    keep = np.isfinite(median)
    x, median = x[:, keep], median[keep]
    x = np.where(np.isfinite(x), x, median)
    mean, scale = x[train].mean(0), x[train].std(0).clip(1e-6)
    metadata = np.array([[r[k] for k in mechanisms] for r in rows]).reshape(len(rows), len(mechanisms))
    batch = np.array([r['batch'] for r in rows])
    for b in np.unique(batch):
        if len(np.unique(split[batch == b])) > 1:
            raise ValueError('Production batches cannot cross train/dev/cal/test partitions')
    if set(split) != {'train', 'dev', 'cal', 'test'}:
        raise ValueError('Require train,dev,cal,test batches')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=((x - mean) / scale).astype(np.float32), y=[int(r['label']) for r in rows],
                        split=split, batch=batch, metadata=metadata, features=np.array(features)[keep])


def tensors(data, ids):
    return torch.tensor(data['x'][ids], dtype=torch.float32), data['metadata'][ids]


def current_incidence(x, metadata, bank, batch, threshold, seed):
    h = incidence(x, metadata, bank)
    rng = np.random.default_rng(seed)
    for b in np.unique(batch):
        rows = np.flatnonzero(batch == b)
        for edge, specification in enumerate(bank):
            if specification['kind'] == 'mechanism':
                continue
            original = h[rows, edge] > 0
            members = np.flatnonzero(original)
            stability = []
            if len(members) >= 2:
                features = x[rows][:, specification['features']]
                for _ in range(10):
                    center = features[rng.choice(members, len(members), replace=True)].mean(0)
                    changed = np.linalg.norm(features - center, axis=1) <= specification['radius']
                    stability.append(np.logical_and(original, changed).sum() / max(1, np.logical_or(original, changed).sum()))
            if not stability or np.mean(stability) < threshold:
                h[rows, edge] = 0
    return h


def predict(model, x, h, frequency, query_x, query_h):
    probability, qp, weight = model(x, h, frequency, query_x, query_h)
    s = relation(h, weight, query_h)
    lower = torch.minimum(torch.maximum(1 - s, probability[None].detach()).min(-1).values, qp.detach())
    upper = torch.maximum(torch.minimum(s, probability[None].detach()).max(-1).values, qp.detach())
    support = (s > 0).any(-1)
    return qp, lower, upper, support, s


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = np.load(args.data)
    train_ids = np.flatnonzero(d['split'] == 'train')
    if len(train_ids) > args.references:
        classes = [train_ids[d['y'][train_ids] == c] for c in [0, 1]]
        chosen = np.concatenate([rng.choice(v, min(len(v), args.references // 2), replace=False) for v in classes])
        train_ids = np.sort(chosen)
    dev_ids = np.flatnonzero(d['split'] == 'dev')
    x, metadata = tensors(d, train_ids)
    vx, vm = tensors(d, dev_ids)
    y = torch.tensor(d['y'][train_ids], dtype=torch.float32)
    if set(y.tolist()) != {0., 1.}:
        raise ValueError('Training references require both common and rare examples')
    residual = x.numpy().copy()
    for b in np.unique(d['batch'][train_ids]):
        selected = d['batch'][train_ids] == b
        residual[selected] -= residual[selected].mean(0)
    bank = build_bank(x.numpy(), metadata, d['y'][train_ids], d['batch'][train_ids], vx.numpy(), vm,
                      d['y'][dev_ids], d['batch'][dev_ids], args, residual)
    h = torch.tensor(incidence(x.numpy(), metadata, bank))
    gates = [e['qualified'] for e in bank]
    if args.variant in ('hgnn', 'random_edges', 'no_qualification'):
        gates = np.ones(len(bank), dtype=bool)
    if args.variant == 'data_only':
        gates = [g and e['kind'] == 'data' for g, e in zip(gates, bank)]
    if args.variant == 'random_edges':
        h = h[torch.randperm(len(h))]
    if args.variant == 'mlp':
        gates = np.zeros(len(bank), dtype=bool)
    frequency = torch.tensor([1 / np.sum(d['batch'][train_ids] == b) for b in d['batch'][train_ids]], dtype=torch.float32)
    model = Hypergraph(x.shape[1], [e['weight'] for e in bank], gates)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best, history = -1., []
    begin = time.perf_counter()
    tracemalloc.start()
    for epoch in range(args.epochs):
        probability, _, weight = model(x, h, frequency)
        loss_vector = nn.functional.binary_cross_entropy(probability, y, reduction='none')
        balanced = torch.where(y > 0, .5 / y.sum().clamp_min(1), .5 / (1 - y).sum().clamp_min(1))
        loss = (balanced * loss_vector).sum()
        _, s = relation(h, weight)
        lower, upper = rough_bounds(s, probability.detach())
        if args.variant not in ('no_rough', 'mlp', 'hgnn'):
            loss += .1 * (1 - lower[y > 0]).mean() + .1 * ((upper - lower).clamp_max(.5) * loss_vector).mean()
        loss += .001 * weight.sum() + .1 * (probability - y).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            query_h = torch.tensor(incidence(vx.numpy(), vm, bank))
            if args.variant == 'random_edges':
                query_h = h[torch.cdist(vx, x).argmin(1)]
            qp = model(x, h, frequency, vx, query_h)[1].numpy()
            score = average_precision_score(d['y'][dev_ids], qp)
        history.append({'epoch': epoch + 1, 'dev_auprc': float(score), 'loss': float(loss.detach())})
        if score > best:
            best = score
            torch.save({'state': model.state_dict(), 'bank': bank, 'x': x, 'h': h, 'frequency': frequency,
                        'gates': list(map(bool, gates)), 'variant': args.variant, 'stability_threshold': args.stability,
                        'seed': args.seed}, out / 'model.pt')
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    (out / 'training.json').write_text(json.dumps({'history': history, 'seconds': time.perf_counter() - begin,
                                                'peak_python_memory_bytes': peak}, indent=2))


def evaluate(args):
    d = np.load(args.data)
    saved = torch.load(args.model, weights_only=False, map_location='cpu')
    bank, x, h, frequency = saved['bank'], saved['x'], saved['h'], saved['frequency']
    model = Hypergraph(x.shape[1], [e['weight'] for e in bank], saved['gates'])
    model.load_state_dict(saved['state'])
    model.eval()
    parts = {}
    with torch.no_grad():
        for partition in ['cal', 'test']:
            ids = np.flatnonzero(d['split'] == partition)
            qx, qm = tensors(d, ids)
            query_incidence = incidence(qx.numpy(), qm, bank)
            if saved['variant'] in ('full', 'data_only', 'no_rough'):
                query_incidence = current_incidence(qx.numpy(), qm, bank, d['batch'][ids], saved['stability_threshold'], saved['seed'])
            if saved['variant'] == 'random_edges':
                query_incidence = h[torch.cdist(qx, x).argmin(1)].numpy()
            qh = torch.tensor(query_incidence)
            qp, lower, upper, support, s = predict(model, x, h, frequency, qx, qh)
            parts[partition] = (ids, qp.numpy(), lower.numpy(), upper.numpy(), support.numpy(), qh)
        ids, p, lo, hi, support, qh = parts['cal']
        y = d['y'][ids]
        if set(y) != {0, 1}:
            raise ValueError('Calibration requires both common and rare labels')
        features = np.column_stack([np.abs(p - .5), hi - lo])
        if saved['variant'] in ('mlp', 'hgnn', 'no_rough'):
            features[:, 1] = 0
        errors = ((p >= .5) != y).astype(int)
        calibrator = LogisticRegression().fit(features, errors) if len(np.unique(errors)) > 1 else float(errors.mean())
        cal_risk = calibrator.predict_proba(features)[:, 1] if not isinstance(calibrator, float) else np.full(len(p), calibrator)
        risk_threshold = float(np.quantile(cal_risk, .9))
        alarm_threshold = float(np.quantile(p[y == 0], .95, method='higher'))
        lower_threshold = float(np.quantile(lo[y == 1], .1)) if (y == 1).any() else .5
        ids, p, lo, hi, support, qh = parts['test']
        y = d['y'][ids]
        features = np.column_stack([np.abs(p - .5), hi - lo])
        if saved['variant'] in ('mlp', 'hgnn', 'no_rough'):
            features[:, 1] = 0
        risk = calibrator.predict_proba(features)[:, 1] if not isinstance(calibrator, float) else np.full(len(p), calibrator)
        prediction = p >= alarm_threshold
        acceptance = support & (risk <= risk_threshold) & (~prediction | (lo >= lower_threshold))
        if saved['variant'] in ('mlp', 'hgnn', 'no_rough'):
            acceptance = risk <= risk_threshold
        ece = 0.
        for low in np.arange(0, 1, .1):
            selected = (p >= low) & (p < low + .1 + 1e-12)
            if selected.any():
                ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
        explanations = []
        base_p, _, weights = model(x, h, frequency)
        for i in range(len(ids)):
            changes = []
            for edge in torch.nonzero(qh[i] * weights > 0, as_tuple=True)[0]:
                changed = weights.clone()
                changed[edge] = 0
                relation_without = relation(h, changed, qh[i:i + 1])
                bound = min(float(torch.maximum(1 - relation_without, base_p[None]).min()), float(p[i]))
                changes.append({'edge': int(edge), 'lower_delta_removed': bound - float(lo[i])})
            explanations.append({'index': int(ids[i]), 'probability': float(p[i]), 'lower': float(lo[i]), 'upper': float(hi[i]),
                                 'supported': bool(support[i]), 'accepted': bool(acceptance[i]), 'edges': changes})
    result = {'rare_auprc': average_precision_score(y, p), 'rare_recall_at_validation_far5': float(prediction[y == 1].mean()),
              'common_false_alarm_rate': float(prediction[y == 0].mean()), 'macro_f1': f1_score(y, prediction, average='macro'),
              'ece': float(ece), 'coverage': float(acceptance.mean()),
              'selective_error': float((prediction[acceptance] != y[acceptance]).mean()) if acceptance.any() else None,
              'bootstrap_hyperedge_jaccard': float(np.mean([e['stability'] for e in bank])),
              'qualified_edges': int(sum(saved['gates']))}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    (out / 'decisions.json').write_text(json.dumps(explanations, indent=2))
    plt.scatter(hi - lo, risk, c=y, alpha=.4)
    plt.xlabel('Rough boundary width')
    plt.ylabel('Calibrated error risk')
    plt.tight_layout()
    plt.savefig(out / 'boundary.png', dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--variant', choices=['full', 'mlp', 'hgnn', 'random_edges', 'data_only', 'no_rough', 'no_qualification'], default='full')
    for key, default in [('epochs', 200), ('clusters', 8), ('bootstrap', 30), ('permutations', 30), ('references', 2000), ('seed', 0)]:
        p.add_argument('--' + key, type=int, default=default)
    for key, default in [('maximum-fraction', .2), ('stability', .5), ('information', .001)]:
        p.add_argument('--' + key, type=float, default=default)
    p = sub.add_parser('evaluate')
    for key in ['data', 'model', 'output']:
        p.add_argument('--' + key, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.command](args)
