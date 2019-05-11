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
    }
}
