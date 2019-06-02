using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Xml;
using Microsoft.ML;

namespace poker_estimator
{
    class Program
    {
        private static string AppPath =>
            Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        
        private static string TrainDataInputPath =>
            Path.Combine(AppPath, "..", "..", "..", "Data", "jira.xml");

        private static MLContext _mlContext;
        private static PredictionEngine<JiraIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;

        private static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            var fromXml = LoadXml();
            _trainingDataView = _mlContext.Data.LoadFromEnumerable(fromXml);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView,
                                                      pipeline);
        }

        private static IEnumerable<JiraIssue> LoadXml()
        {
            var trainingData = new XmlDocument();
            trainingData.Load(TrainDataInputPath);
            return trainingData.GetElementsByTagName("item").Cast<XmlNode>()
                .Select(item => new JiraIssue
                {
                    Key = GetValue(item, "key"),
                    Id = GetAttribute(GetChild(item, "key"), "id"),
                    Title = GetValue(item, "title"),
                    Description = GetValue(item, "description"),
                    Type = GetValue(item, "type"),
                    Time = new SecondPokerizer(GetAttribute(GetChild(item, "timespent"), "seconds")).ToPokerDays(),
                    //TODO more fields
                }).ToList();
        }

        private static string GetAttribute(XmlNode node, string attributeKey)
        {
            return node.Attributes.Cast<XmlAttribute>()
                .Where(attribute => attribute.Name == attributeKey)
                .Select(attribute => attribute.Value)
                .DefaultIfEmpty(null)
                .FirstOrDefault();
        }

        private static XmlNode GetChild(XmlNode item, string nodeName)
        {
            return item.ChildNodes.Cast<XmlNode>()
                .Where(child => child.Name == nodeName)
                .DefaultIfEmpty(null)
                .FirstOrDefault();
        }

        private static string GetValue(XmlNode item, string nodeName)
        {
            return GetChild(item, nodeName)?.InnerText;
        }

        private static IEstimator<ITransformer> ProcessData()
        {
            return _mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "Time", outputColumnName: "PokerValue")
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Type",
                    outputColumnName: "TypeFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                            inputColumnName: "Title",
                            outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Description",
                    outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate(
                    "Features", "TypeFeaturized", "TitleFeaturized", "DescriptionFeaturized"))
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
            var issue = new JiraIssue
            {
                Type = "Change request",
                Title = "WebSphere Upgrade from 9.0 to OpenLiberty",
                Description = "Upgrade all dependencies"
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Time} ===============");
            return trainingPipeline;
        }

    }
}
