# Project on Cancer Cell Metastasis
This Github Repository is intended collect all the codes developed to model the dynamics of a metastatic cell.
The general information on the models used for this purpose is listed below:
  - c($\vec{x}$,t): oxygen consumption field
  - $c_{max}$: maximum oxygen available befor hypoxia
  - C = $\dfrac{c}{c_{max}}$: normalized oxygen consumption ratio
  - $\phi$($\vec{x}$,t): pro-angiogenic factor field
  - $\dot{d}$ = $\alpha$ * C: cell death probability 
  - $\dot{b}$ = $\beta$ * (1 + $\gamma$ - C): cell division probability 
  - R = $\dfrac{\dot{b}}{\dot{d}}$: ratio of cell division and death probabilities (for fractal nature)

Main contributors:
- [Bruno Salgado](https://brunosalgado.website/)
