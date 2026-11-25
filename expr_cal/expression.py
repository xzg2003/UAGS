from abc import ABCMeta, abstractmethod
from typing import List, Type, Union, Tuple
import sys
sys.path.append(".")
from expr_cal.maybe import Maybe, some, none
from expr_cal.stock_data import Stock_Data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
_ExprOrFloat = Union["Expression", float]
_DTimeOrInt = Union["DeltaTime", int]


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: Stock_Data) -> pd.DataFrame: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: _ExprOrFloat) -> "Add": return Add(self, other)
    def __radd__(self, other: float) -> "Add": return Add(other, self)
    def __sub__(self, other: _ExprOrFloat) -> "Sub": return Sub(self, other)
    def __rsub__(self, other: float) -> "Sub": return Sub(other, self)
    def __mul__(self, other: _ExprOrFloat) -> "Mul": return Mul(self, other)
    def __rmul__(self, other: float) -> "Mul": return Mul(other, self)
    def __truediv__(self, other: _ExprOrFloat) -> "Div": return Div(self, other)
    def __rtruediv__(self, other: float) -> "Div": return Div(other, self)
    def __pow__(self, other: _ExprOrFloat) -> "Pow": return Pow(self, other)
    def __rpow__(self, other: float) -> "Pow": return Pow(other, self)
    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(0., self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    @abstractmethod
    def is_featured(self) -> bool: ...


class Feature(Expression):
    def __init__(self, feature:str) -> None:
        self._feature = feature

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        #print(self._feature)
        return pd.DataFrame(data.data[self._feature])

    def __str__(self) -> str: return self._feature

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self.value = value

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        length = len(data.data)
        return pd.DataFrame(np.full((length,1), self.value,dtype=np.float32))

    def __str__(self) -> str: return str(self.value)

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return f"{self._delta_time}"

    @property
    def is_featured(self): return False


def _into_expr(value: _ExprOrFloat) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)


def _into_delta_time(value: Union[int, DeltaTime]) -> DeltaTime:
    return value if isinstance(value, DeltaTime) else DeltaTime(value)


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type["Operator"]: ...

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *args) -> Maybe[str]: ...

    @classmethod
    def _check_arity(cls, *args) -> Maybe[str]:
        arity = cls.n_args()
        if len(args) == arity:
            return none(str)
        else:
            return some(f"{cls.__name__} expects {arity} operand(s), but received {len(args)}")

    @classmethod
    def _check_exprs_featured(cls, args: list) -> Maybe[str]:
        any_is_featured: bool = False
        for i, arg in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                return some(f"{arg} is not a valid expression")
            if isinstance(arg, DeltaTime):
                return some(f"{cls.__name__} expects a normal expression for operand {i + 1}, "
                            f"but got {arg} (a DeltaTime)")
            any_is_featured = any_is_featured or (isinstance(arg, Expression) and arg.is_featured)
        if not any_is_featured:
            if len(args) == 1:
                return some(f"{cls.__name__} expects a featured expression for its operand, "
                            f"but {args[0]} is not featured")
            else:
                return some(f"{cls.__name__} expects at least one featured expression for its operands, "
                            f"but none of {args} is featured")
        return none(str)

    @classmethod
    def _check_delta_time(cls, arg) -> Maybe[str]:
        if not isinstance(arg, (DeltaTime, int)):
            return some(f"{cls.__name__} expects a DeltaTime as its last operand, but {arg} is not")
        return none(str)

    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({','.join(str(op) for op in self.operands)})"


class UnaryOperator(Operator):
    def __init__(self, operand: _ExprOrFloat) -> None:
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls): return UnaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0]]))

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        return self._apply(self._operand.evaluate(data))

    @abstractmethod
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: ...

    @property
    def operands(self): return self._operand,

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return BinaryOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(lambda: cls._check_exprs_featured([args[0], args[1]]))

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        return self._apply(self._lhs.evaluate(data), self._rhs.evaluate(data))

    @abstractmethod
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: ...

    def __str__(self) -> str: return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def operands(self): return self._lhs, self._rhs

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._operand = _into_expr(operand)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls): return RollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0]])
        ).or_else(
            lambda: cls._check_delta_time(args[1])
        )

    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        values = self._operand.evaluate(data)
        result = []
        for index in data.id_slice:
            result.append(self._apply(values.iloc[index],self._delta_time))    
        result = pd.concat(result, axis=0, ignore_index=True)        
        return result                                  

    @abstractmethod
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: ...

    @property
    def operands(self): return self._operand, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self, lhs: _ExprOrFloat, rhs: _ExprOrFloat, delta_time: _DTimeOrInt) -> None:
        self._lhs = _into_expr(lhs)
        self._rhs = _into_expr(rhs)
        self._delta_time = _into_delta_time(delta_time)._delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls): return PairRollingOperator

    @classmethod
    def validate_parameters(cls, *args) -> Maybe[str]:
        return cls._check_arity(*args).or_else(
            lambda: cls._check_exprs_featured([args[0], args[1]])
        ).or_else(
            lambda: cls._check_delta_time(args[2])
        )


    def evaluate(self, data: Stock_Data) -> pd.DataFrame:
        lhs = self._lhs.evaluate(data)
        rhs = self._rhs.evaluate(data)
        result = []
        for index in data.id_slice:
            result.append(self._apply(lhs.iloc[index],rhs.iloc[index], self._delta_time))    
        result = pd.concat(result, axis=0, ignore_index=True)        
        return result                                 

    @abstractmethod
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame, delta_time: int) -> pd.DataFrame: ...

    @property
    def operands(self): return self._lhs, self._rhs, DeltaTime(self._delta_time)

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(np.abs(operand))


class Sign(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(np.sign(operand))


class Log(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(np.log(operand))


class CSRank(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame:
        nan_mask = operand.isnan()
        n = (~nan_mask).sum(dim=1, keepdim=True)
        rank = operand.argsort().argsort() / n
        rank[nan_mask] = pd.nan
        return rank

class Exp(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(np.exp(operand))

class Sqrt(UnaryOperator):
    def _apply(self, operand: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(np.sqrt(operand))
    
class Add(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(lhs.values + rhs.values)


class Sub(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(lhs.values - rhs.values)


class Mul(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(lhs.values * rhs.values)


class Div(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(lhs.values / rhs.values)


class Pow(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        return pd.DataFrame(lhs.values ** rhs.values)


class Greater(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        # Element-wise maximum
        return pd.DataFrame(np.maximum(lhs.values, rhs.values))


class Less(BinaryOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame) -> pd.DataFrame: 
        # Element-wise minimum
        return pd.DataFrame(np.minimum(lhs.values, rhs.values))


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        # This is just for fulfilling the RollingOperator interface
        return operand.shift(delta_time)


class Mean(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).mean()


class Sum(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).sum()


class Std(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).std()


class Var(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).var()


class Skew(RollingOperator):
    # skew = m3 / m2^(3/2)
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def skew_func(x):
            return ((x - x.mean()) ** 3).mean() / (x.std() ** 3)
        return operand.rolling(window=delta_time).apply(skew_func, raw=True)


class Kurt(RollingOperator):
    # kurt = m4 / var^2 - 3
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def kurt_func(x):
            return ((x - x.mean()) ** 4).mean() / (x.var() ** 2) - 3
        return operand.rolling(window=delta_time).apply(kurt_func, raw=True)


class Max(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).max()


class Min(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).min()


class Med(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame: 
        return operand.rolling(window=delta_time).median()


class Mad(RollingOperator):
     def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def mad_func(x):
            return (np.abs(x - x.mean())).mean()
        return operand.rolling(window=delta_time).apply(mad_func, raw=True)


class Rank(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        # Calculate rank for each column independently
        def rank_func(x):
            n = len(x)
            last = x.iloc[-1]
            left = (last < x).sum()
            right = (last == x).sum()
            return (right + left ) / n  # Percentile rank
        return operand.rolling(window=delta_time).apply(rank_func, raw=False)


class Delta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        # Calculate the difference between the current value and the value `delta_time` steps ago
        return operand.diff(periods=delta_time)


class WMA(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def wma_func(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
        return operand.rolling(window=delta_time).apply(wma_func, raw=True)


class EMA(RollingOperator):
    def _apply(self, operand: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def ema_func(x):
            n = len(x)
            alpha = 1 - 2 / (n + 1)
            power = np.arange(n, 0, -1)
            weights = alpha ** power
            weights /= weights.sum()
            return (weights * x).sum()
        return operand.rolling(window=delta_time).apply(ema_func, raw=True)


class Cov(PairRollingOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def cov_func(x, y):
            n = len(x)
            return ((x - x.mean()) * (y - y.mean())).sum() / (n - 1)
        
        result = []
        for i in range(len(lhs)):
            if i < delta_time:
                result.append(np.nan)
                continue
            lhs_data = lhs.iloc[i-delta_time:i].values
            rhs_data = rhs.iloc[i-delta_time:i].values
            result.append(cov_func(lhs_data, rhs_data))
        return pd.DataFrame(result)


class Corr(PairRollingOperator):
    def _apply(self, lhs: pd.DataFrame, rhs: pd.DataFrame, delta_time: int) -> pd.DataFrame:
        def corr_func(x, y):
            n = len(x)
            return ((x-x.mean()) * (y - y.mean())).sum() / (n * x.std() * y.std())
        
        result = []
        for i in range(len(lhs)):
            if i < delta_time:
                result.append(np.nan)
                continue
            lhs_data = lhs.iloc[i-delta_time:i].values
            rhs_data = rhs.iloc[i-delta_time:i].values
            result.append(corr_func(lhs_data, rhs_data))
        return pd.DataFrame(result)


Operators: List[Type[Operator]] = [
    # Unary
    Abs, Sign, Log, CSRank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min,
    Med, Mad, Rank, Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]
'''
def cal_ic(factor_values: pd.DataFrame, target_values: pd.DataFrame, data: Future_Data) -> float:
    # 计算每个品种平均IC -- robustly handle NaNs and constant arrays
    ic_values = []
    if not isinstance(factor_values, pd.DataFrame):
        factor_values = pd.DataFrame(factor_values, columns=['factor'])
    if not isinstance(target_values, pd.DataFrame):
        target_values = pd.DataFrame(target_values, columns=['target'])

    invalid = np.isnan(factor_values) | ~np.isfinite(factor_values)
    factor_values[invalid] = 0
    target_values = target_values.fillna(0)
    for idx_slice in data.test_slice:
        f = np.array(factor_values.iloc[idx_slice]).ravel()
        t = np.array(target_values.iloc[idx_slice]).ravel()
        # need at least two non-constant samples for Pearson; otherwise result is NaN
        
        if len(f) < 2 :
            ic = 0
        else:
            ic = pearsonr(f, t)[0]
        
        if np.isnan(ic) or np.isinf(ic):
            ic_values.append(0)
        else:
            ic_values.append(ic)
        #print(ic)
    return np.mean(ic_values)'''

if __name__=="__main__":
    data = Stock_Data(stock_list='csi300', k_line='1d', start_date='2020-01-01', end_date='2022-12-31')
    #data = Future_Data(insrument_list=['A','AG'],k_line='1d', start_date='2020-01-01', end_date='2020-01-10')
    open = Feature('open')
    close = Feature('close')
    high = Feature('high')
    low = Feature('low')
    alpha = (Sub(Div(Mul(10.0,Sum(low,20)),high),10.0)).evaluate(data)
    #alpha = Exp(Sqrt(high)).evaluate(data)
    target = ((Ref(close, -1) - close)/close * 1000).evaluate(data)
    target1 = data.data['FCT1'].values
    #cal_ic(target1, target, data)
    print(cal_ic(target1, target, data))
    