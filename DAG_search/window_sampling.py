"""Sample explicit window lengths in symbolic UDFS expressions."""
import ast

from DAG_search.config import ROLLING_DICT


def sample_windows(expression, windows, rng):
    """Sample each implicit rolling call once and return an explicit expression."""
    class Sampler(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in ROLLING_DICT:
                if len(node.args) not in (1, 2) or node.keywords:
                    raise ValueError(f"Expected one or two arguments for {node.func.id}")
                node.func.id = ROLLING_DICT[node.func.id]
                window = ast.Constant(value=int(rng.choice(windows)))
                if len(node.args) == 1:
                    node.args.append(window)
                else:
                    node.args[1] = window
            return node

    tree = Sampler().visit(ast.parse(expression, mode="eval"))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
