Troubleshooting
================

Having trouble getting the model to work? Here are some ideas that you could try.

* You could remove the priors over the generation interval and delay parameters. This can be done by modifying the build dictionary returned by ``EpidemiologicalParameters()``. e.g., you could do ``bd = ep.get_model_build_dict()`` and ``bd['gi_mean_sd']=0`` to set remove the prior over the generation interval mean. There are also priors over the delay between infection and case confirmation, as well as infection to death.

* You need to make sure all invalid data (daily cases and deaths in each region) is masked. The output distribution we use is a Negative Binomial, which is a discrete distribution. PyMC3, by default, will attempt to impute missing values. Since these missing values are discrete, PyMC3 uses the Metropolis algorithm that does not perform well. This can throw off the results significantly. The model should **only be using NUTS, not Metropolis**.

* Another thing you could try is switching to a fixed-effects model (i.e., ``DefaultModel()``)) to see whether that fixes your issue.

You can also email Mrinank at ``mrinank [at] robots [dot] ox [dot] ac [dot] uk`` if you have any additional questions.

