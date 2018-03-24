'''
Feature Engineering:
- Cleaning and processing raw input data into useful format that can be effective feature vector for training models.

Heuristics for Feature Engineering:
- Real value input data can be used straight away as a feature.
- String value can be encoded as a one hot value \
  (a binary value to represent if the string value exists)
  (so there will be one feature for each possible value)
  (end up with a binary vector)
- a feature is useful if it has many occurences of non-zero or null feature values
- a feature should have clear and obvious meaning
- a feature should not be encoded to take "magic" or arbitrary value
  (instead use an indicator value or one hot encoding etc)
  (for Categorical value, similarly use one hot encoding instead of numerical encoding)
- feature value should not change over time to train an effective model
  (be careful that upstream system that generates the data may change the value over time)
- feature should not have extreme outlier value that will disrupt the model training
- Use the Binning Trick:
  (split feature values into bin to create a more helpful predictor)
- Possible workflow:
  VISUALISE - histograms etc to quickly detect problematic feature values
  DEBUG - clean up and process the feature values
  MONITOR - design ways of monitoring the stability of features over time as we continuously train our model
    (can use key statistics to help with detection, e.g. Max, Min, Mean, Median, Std Dev)
  
- Scaling values, to prevent different features that exist on different scales from slowing down the training unnecessarily
  LINEAR MAPPING: map all values between min and max of feature to [-1,+1]
  Z SCORE: map all value using the formula: scaledvalue = (value - mean)/stddev
  LOGARITHMIC SCALING: take the Logarithm of feature values to scale wide spread data
  CLIPPING VALUES: clipping values to a ceiling or a floor will prevent outlier effects but may artificially skew the result
- Scrubbing
  - Remove omitted values or incomplete data
  - Remove duplicated records
  - Correcting bad labels due to entry errors
  - Correcting bad feature values due to entry errors
'''
