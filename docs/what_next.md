
## Overview of the Current Project
As described in the `readme`, the project `Causality Exploration` currently provides a foundational framework for delving into the realm of causality using the `Causalink` package. It also offers a practical, yet simple and straightforward, approach for utilizing this framework in prediction tasks through the `Causalgo` package.

## Follow-up Ideas
Considering that AI-vidence aims to elucidate models based on our initial `regional approach`, which is presently being developed within the `AntakIA` package, causality can potentially augment this approach. This can be achieved by:
- Integrating causal graphs and the warning changes into the AntakIA GUI framework to enhance the visualization of interactions between features, both on a global scale and within specific regions and alert in case of evolution of the given space values.
- Employing The CausAlgo algorithm to redefine the regionalization process. One concept is to experiment with the creation of a new explanatory space (as opposed to the current usage of SHAP) founded on actual causal connections, rather than simple correlations. The preliminary approach in Causalgo involves identifying the moment *when the variable becomes involved during the back process* for each instance, at which point the non-coherence is disrupted. The aim is to construct a vector, either binary or weighted, that can describe this explanation. While this may appear complex at present, it holds significant promise.
- Integrate causal and knowledge graphs within the Sofia framework for explanability of LLMs 
