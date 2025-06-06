from typing import Union, Tuple, List
from dataclasses import dataclass, field


@dataclass
class Profile:
  name: Union[str, None]=None
  mobility_constr: bool=False
  location: Union[None, Tuple[float, float]]=None
  max_disp: Union[float, None]=None
  avoid: List[str] = field(default_factory=list)
  culture: float=0.5
  nature: float=0.5
  nlife: float=0.5
  local_imp: float=0.5
  co2: float=0.5
