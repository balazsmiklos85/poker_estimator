using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace poker_estimator
{
    class Program
    {
        private static string _appPath =>
            Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath =>
            Path.Combine(_appPath, "..", "..", "..", "Data", "jira.csv");
        private static string _testDataPath =>
            Path.Combine(_appPath, "..", "..", "..", "Data", "jira_test.csv");
        private static string _modelPath =>
            Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<JiraIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<JiraIssue>(
                _trainDataPath,hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView,
                                                      pipeline);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            return _mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Time", outputColumnName: "PokerValue")
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                            inputColumnName: "Summary",
                            outputColumnName: "SummaryFeaturized"))
//                .Append(_mlContext.Transforms.Text.FeaturizeText(
//                            inputColumnName: "Description",
//                            outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate(
                    "Features", "SummaryFeaturized"/*, "DescriptionFeaturized"*/))
                .AppendCacheCheckpoint(_mlContext);
        }

        private static object BuildAndTrainModel(IDataView trainingDataView,
                                                 IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(
                _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    "PokerValue", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                        "PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<JiraIssue, IssuePrediction>(
                _trainedModel);
            JiraIssue issue = new JiraIssue() {
                Summary = "Rewrite the whole application"
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Time} ===============");
            return trainingPipeline;
        }

    }
}
