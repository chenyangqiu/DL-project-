## Experiments for Milestone

#### Important: Chang First！

##### Time: 11.30-12.4

1. make the attention vector varying over time: $a^k$.
2. adaptive stepsize: $\eta^k$.
3. increase the number of feature for attention vector:
   1. function value difference
   2. momentum: difference between current and last iterate
      - neighbor info(x,f)
      - my info(x,f)
   3. degree? may be useful for time-varying network

#### Test for different problem dimension

- dim = 5, 10, 50, 100

#### Test for tranning iteration

- \#iteration= 10,20,50
- Or: for certian optimial distance error: 10^-3, 10^-4, 10^-5-> require how many iterations？

#### Test the different combination of additional feature

#### Test the different size and types of graphs

- size: num_of_nodes: 10,50,100,500; keep connectivity ratio: 

- Check this: https://networkx.org/documentation/stable/reference/generators.html
- Current: we only focus on undirected graphs.

#### Increase the number of epoch:

- \#epochs: 100,200,300

## Writing for Milestone



| Parts                | Time       | Assign                 |
| -------------------- | ---------- | ---------------------- |
| Frame for report     | 11.30-12.3 | He Wang                |
| Intro + related work | 12.3-12.7  | He Wang                |
| Problem statement    | 12.3-12.7  | He Wang                |
| Technical approach   | 12.4-12.7  | Better in English      |
| Intermediate results | 12.4-12.8  | Who run the experiment |
