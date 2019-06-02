using System;
using System.Collections.Generic;
using System.Linq;
using Atlassian.Jira;
using Microsoft.ML;
using RestSharp.Extensions;

namespace poker_estimator
{
    public class ThreePointPokerPredictor
    {
        private List<PredictionEngine<JiraIssue, IssuePrediction>> _predictionEngines;

        public ThreePointPokerPredictor(MLContext mlContext, IDataView trainingDataView,
            IEstimator<ITransformer> pipeline)
        {
            _predictionEngines = new List<PredictionEngine<JiraIssue, IssuePrediction>>();
            
            Console.Write("Training SDCA Maxium Entropy model...");
            var trainingPipeline = pipeline.Append(
                    mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                        "PokerValue", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            _predictionEngines.Add(mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(trainedModel));
            Console.WriteLine("OK");
            
            Console.Write("Training Naive Bayes model...");
            trainingPipeline = pipeline.Append(
                    mlContext.MulticlassClassification.Trainers.NaiveBayes(
                        "PokerValue", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));
            trainedModel = trainingPipeline.Fit(trainingDataView);
            _predictionEngines.Add(mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(trainedModel));
            Console.WriteLine("OK");
            
            Console.Write("Training LBFGS Maximum Entropy model...");
            trainingPipeline = pipeline.Append(
                    mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                        "PokerValue", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));
            trainedModel = trainingPipeline.Fit(trainingDataView);
            _predictionEngines.Add(mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(trainedModel));
            Console.WriteLine("OK");
            
            Console.Write("Training SDCA Non-calibrated model...");
            trainingPipeline = pipeline.Append(
                    mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(
                        "PokerValue", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));
            trainedModel = trainingPipeline.Fit(trainingDataView);
            _predictionEngines.Add(mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(trainedModel));
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