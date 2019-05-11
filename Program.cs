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
            Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath =>
            Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath =>
            Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(
                _trainDataPath,hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView,
                                                      pipeline);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            return _mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                            inputColumnName: "Title",
                            outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                            inputColumnName: "Description",
                            outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate(
                    "Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);
        }

        private static object BuildAndTrainModel(IDataView trainingDataView,
                                                 IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(
                _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    "Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                        "PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(
                _trainedModel);
            GitHubIssue issue = new GitHubIssue() {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }

    }
}
