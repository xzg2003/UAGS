"""Select from completed factor_ic.csv and fit native linear combination weights."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import importlib
import json
import hashlib
import operator
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from result_io import PROJECT, METHOD, add_arguments, jobs, digest, update_summary
from combination_engine import add_combination_arguments, validate_arguments, run_combination


def load_calculated_ic(path, market, seed, dates):
    """Require a complete IC file from calculate_ic.py for the same source and dates."""
    path = Path(path)
    metadata_path = path.with_name('ic_metadata.json')
    if not metadata_path.is_file():
        raise ValueError(f'Missing IC metadata: {metadata_path}; run calculate_ic.py first')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not metadata.get('complete') or metadata.get('output_sha256') != digest(path):
        raise ValueError(f'IC output is incomplete or modified: {path}')
    contract = metadata['contract']
    arguments = contract['arguments']
    if arguments['instruments'] != market or int(arguments['seed']) != seed:
        raise ValueError(f'IC market/seed mismatch: {path}')
    if any(arguments[key] != value for key, value in dates.items()):
        raise ValueError(f'IC date ranges differ from combination: {path}')
    source_path = Path(contract['input'])
    if not source_path.is_file() or digest(source_path) != contract['input_sha256']:
        raise ValueError(f'Sampled expressions changed since IC calculation: {source_path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'expr', 'expr_id', 'candidate_id', 'ic_train', 'ic_valid', 'ic_test',
                    'status_train', 'status_valid', 'status_test'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Missing IC fields in {path}')
        rows = list(reader)
    if not rows or len(rows) != metadata['candidates']:
        raise ValueError(f'IC candidate count mismatch: {path}')
    return rows, {'source': str(path), 'source_sha256': digest(path),
                  'expression_source': str(source_path), 'ic_metadata': str(metadata_path)}


def backend():
    if METHOD=='gp':
        stock=importlib.import_module('data.stock_data')
        native=importlib.import_module('data.expressions')
        normalize=importlib.import_module('utils.normalize').normalize_by_day
    else:
        stock=importlib.import_module('expr_cal.stock_data')
        native=importlib.import_module('expr_cal.expression')
        normalize=importlib.import_module('torch_utils.pytorch_utils').normalize_by_day
    return stock,native,normalize


def parse_expression(source,native):
    """Construct local native operators; no Python eval and no foreign operator rewrite."""
    features=dict(zip(('x_0','x_1','x_2','x_3','x_4'),('open','close','high','low','volume')))
    rolling={'mean':'Mean','std':'Std','ref':'Ref','max':'Max','var':'Var','min':'Min','sum':'Sum'}
    arithmetic={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}
    def numeric(node):
        if isinstance(node,ast.Constant):return float(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return numeric(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
        raise ValueError('Expected literal')
    def visit(node):
        if isinstance(node,ast.Name) and node.id in features:return native.Feature(features[node.id])
        if isinstance(node,ast.Name) and node.id in features.values():return native.Feature(node.id)
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return native.Constant(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            return -visit(node.operand) if isinstance(node.op,ast.USub) else visit(node.operand)
        if isinstance(node,ast.BinOp) and type(node.op) in arithmetic:
            return arithmetic[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            name,args=node.func.id,list(node.args)
            if name=='Feature' and len(args)==1:return native.Feature(ast.literal_eval(args[0]))
            if name=='Constant' and len(args)==1:return native.Constant(numeric(args[0]))
            if METHOD=='UDFS_final' and name in rolling:
                name=rolling[name]
                if len(args)==1:args.append(ast.Constant(value=5))
            cls=getattr(native,name,None)
            if not isinstance(cls,type) or not issubclass(cls,native.Operator):raise ValueError('Unknown operator: '+name)
            if issubclass(cls,(native.RollingOperator,native.PairRollingOperator)):
                window=numeric(args.pop())
                if not window.is_integer():raise ValueError('Nonintegral window')
                return cls(*(visit(a) for a in args),int(window))
            return cls(*(visit(a) for a in args))
        raise ValueError('Unsupported expression: '+ast.dump(node))
    return visit(ast.parse(source,mode='eval').body)


class NativeData:
    def __init__(self,args,market,start,end):
        import numpy as np
        stock,self.native,self.normalize=backend()
        if args.data_dir:stock.DATA_DIR=str(args.data_dir.expanduser().resolve())
        self.data=stock.Stock_Data(stock_list=market,k_line='1d',start_date=start,end_date=end)
        close=self.native.Feature('close')
        label=((self.native.Ref(close,-2)/self.native.Ref(close,-1)-1)*1000).evaluate(self.data)
        self.target=self.normalize(label.fillna(0).values,self.data).reshape(-1)
        if not np.isfinite(self.target).all():raise ValueError('Nonfinite native target')
        self.dates=[self.data.data.loc[index,'date'].iloc[0] for index in self.data.test_slice]
    def factor(self,expr):
        import numpy as np
        value=parse_expression(expr,self.native).evaluate(self.data)
        value=np.asarray(value.values if hasattr(value,'values') else value,dtype=float).copy()
        value[~np.isfinite(value)]=0
        value=self.normalize(value,self.data).reshape(-1)
        if value.shape!=self.target.shape or not np.isfinite(value).all():raise ValueError('Invalid native factor')
        return value
    def training_arrays(self,columns):
        import numpy as np
        return np.column_stack(columns), self.target
    def corr(self,left,right):
        import numpy as np
        return float(np.dot(left,right)/self.data.n_days)
    def metrics(self,pred):
        import numpy as np
        from scipy.stats import rankdata
        normalized=self.normalize(pred,self.data)
        daily=[]
        for date,index in zip(self.dates,self.data.test_slice):
            x,y=normalized[index],self.target[index]
            ic=float(np.dot(x,y))
            rx,ry=rankdata(x),rankdata(y)
            rx-=rx.mean();ry-=ry.mean()
            denom=float(np.linalg.norm(rx)*np.linalg.norm(ry))
            ric=float(np.dot(rx,ry)/denom) if denom>1e-12 else 0.0
            daily.append(dict(date=str(date.date()),ic=ic,ric=ric))
        def ir(v):
            sd=np.std(v,ddof=1) if len(v)>1 else 0
            return float(np.mean(v)/sd) if sd>1e-12 else None
        ic=np.array([r['ic'] for r in daily]);ric=np.array([r['ric'] for r in daily])
        return dict(ic=float(ic.mean()),ric=float(ric.mean()),icir=ir(ic),ricir=ir(ric)),daily


def data_signature(root, market):
    root = Path(root).expanduser().resolve()
    files = sorted((root/'1d').glob('*/*.csv'))
    if not files:
        raise ValueError('No stock CSV files in '+str(root))
    state = hashlib.sha256()
    for path in files:
        stat = path.stat()
        state.update(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return dict(root=str(root), stock_file_count=len(files), stock_stats_sha256=state.hexdigest(),
                instruments_sha256=digest(root/(market+'_constituents.csv')))


def completed_combination(directory, requested_counts=None):
    """Recognize a finished canonical combination, not a partial run."""
    config_path, summary_path = directory/'combination_config.json', directory/'summary.csv'
    if not config_path.is_file() or not summary_path.is_file() or \
            not (directory/'selected_idx.csv').is_file():
        return False
    try:
        stored_counts = {int(k) for k in json.loads(config_path.read_text(encoding='utf-8'))['factor_counts']}
        counts = set(requested_counts) if requested_counts is not None else stored_counts
        with summary_path.open(encoding='utf-8-sig', newline='') as handle:
            summary = list(csv.DictReader(handle))
        summary_counts = {int(row['requested_k']) for row in summary}
        if not counts or not counts <= stored_counts or not counts <= summary_counts or len(summary) != len(summary_counts):
            return False
        for k in counts:
            for name in (f'optimizer_{k}.json', f'selected_factors_{k}.csv',
                         f'metrics_{k}.json', f'daily_metrics_{k}.csv', f'ic_{k}.csv'):
                if not (directory/name).is_file():
                    return False
        for row in summary:
            metrics = json.loads((directory/f"metrics_{int(row['requested_k'])}.json").read_text(encoding='utf-8'))
            if any(float(row[key]) != float(metrics[key])
                   for key in ('actual_k', 'test_ic', 'test_ric')):
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def cache_matches_request(directory, args, task):
    """Reject checkpoints built from different inputs, settings or market data."""
    try:
        config = json.loads((directory/'combination_config.json').read_text(encoding='utf-8'))
        source = config['ic_source']
        if config['instruments'] != task['instruments'] or int(config['seed']) != task['seed']:
            return False
        if source['source_sha256'] != digest(task['input']):
            return False
        expression_source = Path(source['expression_source'])
        metadata = json.loads(Path(source['ic_metadata']).read_text(encoding='utf-8'))
        if not expression_source.is_file() or digest(expression_source) != metadata['contract']['input_sha256']:
            return False
        if config['min_valid_ic'] != args.min_valid_ic or \
                config['corr_threshold'] != args.corr_threshold or config['device'] != args.device:
            return False
        for key in ('maxiter', 'maxfun', 'ftol', 'gtol'):
            if config['optimizer'][key] != getattr(args, 'optimizer_' + key):
                return False
        for key in ('train_start','train_end','valid_start','valid_end','test_start','test_end'):
            if config['arguments'][key] != getattr(args, key):
                return False
        stock, _, _ = backend()
        root = args.data_dir.expanduser().resolve() if args.data_dir else Path(stock.DATA_DIR).resolve()
        if config['data_signature'] != data_signature(root, task['instruments']):
            return False
        return all(digest(PROJECT/name) == value for name, value in
                   config['sources_sha256'].items() if not name.startswith('analyze_result/'))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def combine(args, task):
    dates = {key: getattr(args, key) for key in
             ('train_start', 'train_end', 'valid_start', 'valid_end', 'test_start', 'test_end')}
    rows, source = load_calculated_ic(task['input'], task['instruments'], task['seed'], dates)
    stock, _, _ = backend()
    data_root = args.data_dir.expanduser().resolve() if args.data_dir else Path(stock.DATA_DIR).resolve()
    ic_contract = json.loads(Path(source['ic_metadata']).read_text(encoding='utf-8'))['contract']
    current_data_signature = data_signature(data_root, task['instruments'])
    if str(data_root) != ic_contract['arguments']['data_dir'] or \
            current_data_signature != ic_contract['data_signature']:
        raise ValueError('Market data differs from the completed IC calculation')
    native_sources = ('data/stock_data.py','data/expressions.py','utils/normalize.py') if METHOD=='gp' else (
        'expr_cal/stock_data.py','expr_cal/expression.py','torch_utils/pytorch_utils.py')
    sources = ('analyze_result/combine_factors.py','analyze_result/combination_engine.py',
               'analyze_result/result_io.py',*native_sources)
    parameters = {key:getattr(args,key) for key in (
        'train_start','train_end','valid_start','valid_end','test_start','test_end')}
    parameters['data_dir'] = str(data_root)
    provenance = dict(method=METHOD, instruments=task['instruments'], seed=task['seed'],
                      ic_source=source, arguments=parameters,
                      sources_sha256={name:digest(PROJECT/name) for name in sources},
                      data_signature=current_data_signature,
                      normalization='native normalize_by_day: daily zscore/sqrt(stock count)',
                      cached_ic_provenance='complete calculate_ic.py output, verified against expressions, dates and data signature')
    def split(name):
        return NativeData(args, task['instruments'], parameters[name+'_start'], parameters[name+'_end'])
    summaries = run_combination(rows, split('train'), lambda: split('valid'), lambda: split('test'),
                                task['output'], args, provenance)
    summaries = [dict(instruments=task['instruments'], seed=task['seed'], **row) for row in summaries]
    update_summary(args.analysis_root,'combination_summary.csv',summaries,('instruments','seed','requested_k'))
    print(json.dumps(summaries),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    add_combination_arguments(parser)
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute and replace an existing combination result')
    parser.add_argument('--data-dir',type=Path)
    for key,value in zip(('train_start','train_end','valid_start','valid_end','test_start','test_end'),
                         ('2010-01-01','2018-12-31','2019-01-01','2019-12-31','2020-01-01','2021-12-31')):
        parser.add_argument('--'+key.replace('_','-'),default=value)
    args=parser.parse_args(argv)
    validate_arguments(args,parser)
    try:
        dates=[datetime.fromisoformat(getattr(args,k)) for k in ('train_start','train_end','valid_start','valid_end','test_start','test_end')]
    except ValueError:
        parser.error('Dates must use YYYY-MM-DD')
    if not dates[0]<=dates[1]<dates[2]<=dates[3]<dates[4]<=dates[5]:
        parser.error('Dates must be ordered and disjoint')
    tasks=jobs(args,parser,'factor_ic.csv')
    if args.dry_run:
        print(json.dumps(tasks,default=str,indent=2))
        return
    for task in tasks:
        output = Path(task['output'])/'combination'
        if output.exists() and not args.overwrite and not cache_matches_request(output, args, task):
            archived = output.with_name('combination_stale_' +
                                        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            output.rename(archived)
            print(f"{task['instruments']}/{task['seed']}: incompatible cache saved as {archived}; starting over",
                  flush=True)
        if output.exists() and not args.overwrite:
            if completed_combination(output, args.n_factors):
                print(f"{task['instruments']}/{task['seed']}: requested combinations already exist; skipped",
                      flush=True)
                continue
        backup = None
        if output.exists() and args.overwrite:
            if not output.is_dir():
                parser.error(f'Combination output is not a directory: {output}')
            backup = output.with_name('.combination_backup_' +
                                      datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            output.rename(backup)
        try:
            if args.overwrite:
                args.selection_mode = 'recompute'
            combine(args,task)
        except BaseException:
            if backup is not None:
                if output.exists():
                    shutil.rmtree(output)
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup)


if __name__=='__main__':main()
