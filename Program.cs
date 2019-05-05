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
            Console.WriteLine("Hello World!");
        }
    }
}
