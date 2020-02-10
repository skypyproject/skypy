# ADR 2: Mpc or Mpc/h
February 6, 2020

## Context
We need to decide on a unit convention as to whether units include the factor /h or not (for instance Mpc or Mpc/h as a unit of distance). For further discussion see e.g. 10.1017/pasa.2013.31

## Decision Drivers
- Flexibility: Mpc/h allows results to be easily propagated across the 'unknown' value of h (0.68 or 0.74 or something else).
- Consistency / least surprise: the default for astropy is Mpc

## Considered Options
- Mpc
- Mpc/h

## Decision Outcome
After [discussion](https://github.com/skypyproject/skypy/issues/23) and offline, Mpc has been chosen to ensure the closest integration and least surprise for astropy.