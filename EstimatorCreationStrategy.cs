using Microsoft.ML;

namespace poker_estimator
{
    public abstract class EstimatorCreationStrategy<T> where T : class, ITransformer
    {
        protected readonly MLContext MlContext;

        protected EstimatorCreationStrategy(MLContext mlContext, string modelName)
        {
            ModelName = modelName;
            MlContext = mlContext;
        }
        
        public string ModelName { get; }

        public abstract IEstimator<T> Create(string labelColumnName, string featureColumnName);
    }

    public class SdcaMaximumEntropyStrategy : EstimatorCreationStrategy<ITransformer>
    {
        public SdcaMaximumEntropyStrategy(MLContext mlContext) : base(mlContext, "SDCA Maximum Entropy") { }

        public override IEstimator<ITransformer> Create(string labelColumnName, string featureColumnName)
        {
            return MlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName, featureColumnName);
        }
    }
    
    public class NaiveBayesStrategy : EstimatorCreationStrategy<ITransformer>
    {
        public NaiveBayesStrategy(MLContext mlContext) : base(mlContext, "Naive Bayes")
        {
            
        }

        public override IEstimator<ITransformer> Create(string labelColumnName, string featureColumnName)
        {
            return MlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName, featureColumnName);
        }
    }
    
    public class SdcaNonCalibratedStrategy : EstimatorCreationStrategy<ITransformer>
    {
        public SdcaNonCalibratedStrategy(MLContext mlContext) : base(mlContext, "SDCA Non-calibrated") { }

        public override IEstimator<ITransformer> Create(string labelColumnName, string featureColumnName)
        {
            return MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(labelColumnName, featureColumnName);
        }
    }

    public class LbfgsMaximumEntropyStrategy : EstimatorCreationStrategy<ITransformer>
    {
        public LbfgsMaximumEntropyStrategy(MLContext mlContext) :base(mlContext, "LBFGS Maximum Entropy") { }

        public override IEstimator<ITransformer> Create(string labelColumnName, string featureColumnName)
        {
            return MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName, featureColumnName);
        }
    }
}