using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace poker_estimator
{
    public class ThreePointPokerPredictor
    {
        private List<PredictionEngine<JiraIssue, IssuePrediction>> _predictionEngines;
        private MLContext _mlContext;
        private IDataView _trainingDataView;
        private IDataView _testDataView;
        private IEstimator<ITransformer> _pipeline;

        public ThreePointPokerPredictor(MLContext mlContext, IReadOnlyCollection<JiraIssue> trainingData,
            IEstimator<ITransformer> pipeline)
        {
            _mlContext = mlContext;
            _pipeline = pipeline;
            SetUpTestData(trainingData);
            
            _predictionEngines = new List<PredictionEngine<JiraIssue, IssuePrediction>>();

            TransformerChain<KeyToValueMappingTransformer> model;
            model = CreatePredictionEngine(new NaiveBayesStrategy(mlContext));
            TestPredictionEngine(model);
            model = CreatePredictionEngine(new LbfgsMaximumEntropyStrategy(mlContext));
            TestPredictionEngine(model);
            model = CreatePredictionEngine(new SdcaMaximumEntropyStrategy(mlContext));
            TestPredictionEngine(model);
            model = CreatePredictionEngine(new SdcaNonCalibratedStrategy(mlContext));
            TestPredictionEngine(model);
        }

        private void SetUpTestData(IReadOnlyCollection<JiraIssue> allData)
        {
            var size = allData.Count;
            var testSize = size <= 64 ? size / 2 : size / 20;
            var rng = new Random();
            var randomized = allData.OrderBy(d => rng.Next()).ToList();
            var trainingData = randomized.Take(size - testSize).ToList();
            var testData = randomized.TakeLast(testSize).ToList();
            _trainingDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            _testDataView = _mlContext.Data.LoadFromEnumerable(testData);
        }

        private TransformerChain<KeyToValueMappingTransformer> CreatePredictionEngine(
            EstimatorCreationStrategy<ITransformer> estimatorStrategy)
        {
            Console.WriteLine($"== {estimatorStrategy.ModelName} model ==");
            Console.Write("Training model...");
            var estimator = estimatorStrategy.Create("PokerValue", "Features");
            var trainingPipeline = _pipeline.Append(estimator)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));
            var trainedModel = trainingPipeline.Fit(_trainingDataView);
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(trainedModel);
            _predictionEngines.Add(predictionEngine);
            Console.WriteLine("OK");
            return trainedModel;
        }
        
        private void TestPredictionEngine(ITransformer model)
        {
            var prediction = model.Transform(_testDataView);
            var metrics = _mlContext.MulticlassClassification.Evaluate(prediction, labelColumnName: "PokerValue");
            Console.WriteLine("Confusion matrix:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"Micro accuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Log loss: {metrics.LogLoss}");
            Console.WriteLine($"Log loss reduction: {metrics.LogLossReduction}");
            Console.WriteLine($"Per class log loss: [{string.Join(", ", metrics.PerClassLogLoss)}]");
            Console.WriteLine($"Top K accuracy: {metrics.TopKAccuracy}");
            Console.WriteLine($"Top K prediction count: {metrics.TopKPredictionCount}");
        }

        public ThreePointEstimation Predict(JiraIssue issue)
        {
            var results = _predictionEngines.Select(predictionEngine => predictionEngine.Predict(issue))
                .Select(p => decimal.Parse(p.Time))
                .ToList();
            var result = new ThreePointEstimation
            {
                Min = results.Min(),
                Max = results.Max(),
                Mean = results.Average()
            };
            result.Estimate();
            return result;
        }
    }
}