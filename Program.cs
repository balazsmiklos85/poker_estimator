﻿using System;
using System.Collections.Generic;
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
        private static string DataInputPath =>
            Path.Combine(AppPath, "..", "..", "..", "Data", "to_estimate.xml");

        private static MLContext _mlContext;
        private static ThreePointPokerPredictor _predictionEngine;

        private static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);
            Console.WriteLine("=== Training ===");
            var fromXml = LoadXml(TrainDataInputPath).ToList();
            var pipeline = ProcessData();
            _predictionEngine = new ThreePointPokerPredictor(_mlContext, fromXml, pipeline);

            Console.WriteLine("=== Estimation ===");
            var toEstimate = LoadXml(DataInputPath);
            foreach (var issue in toEstimate)
            {
                var prediction = _predictionEngine.Predict(issue);
                Console.WriteLine($"{issue.Key}: {prediction.Min}/{prediction.Mean}/{prediction.Max} = {prediction.Time}");
            }
        }

        private static IEnumerable<JiraIssue> LoadXml(string path)
        {
            Console.WriteLine($"Loading file: {path}");
            var trainingData = new XmlDocument();
            trainingData.Load(path);
            var result = trainingData.GetElementsByTagName("item").Cast<XmlNode>()
                .Select(item => new JiraIssue
                {
                    Key = GetValue(item, "key"),
                    Id = GetAttribute(GetChild(item, "key"), "id"),
                    ParentId = GetAttribute(GetChild(item, "parent"), "id"), 
                    Title = GetValue(item, "title"),
                    Description = GetValue(item, "description"),
                    Type = GetValue(item, "type"),
                    Time = new SecondPokerizer(GetAttribute(GetChild(item, "timespent"), "seconds")).ToPokerDays(),
                    Environment = GetValue(item, "environment"),
                    Reporter = GetValue(item, "reporter"),
                    Version = GetValue(item, "version"),
                    Priority = GetValue(item, "priority"),
                    OriginalEstimate = new SecondPokerizer(GetAttribute(GetChild(item, "timeestimate"), "seconds")).ToPokerDays(),
                    CreatedTime = GetValue(item, "created"),
                    //TODO more fields
                }).ToList();
            foreach (var issue in result)
            {
                var parent = result.Where(r => r.Id == issue.ParentId && !string.IsNullOrEmpty(issue.ParentId))
                    .DefaultIfEmpty(new JiraIssue())
                    .FirstOrDefault();
                issue.ParentTitle = parent?.Title;
                issue.ParentDescription = parent?.Description;
            }
            Console.WriteLine($"{result.Count} issues loaded");
            return result;
        }

        private static string GetAttribute(XmlNode node, string attributeKey)
        {
            return node?.Attributes.Cast<XmlAttribute>()
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
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Environment",
                    outputColumnName: "EnvironmentFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Reporter",
                    outputColumnName: "ReporterFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Version",
                    outputColumnName: "VersionFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "Priority",
                    outputColumnName: "PriorityFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "OriginalEstimate",
                    outputColumnName: "OriginalEstimateFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "CreatedTime",
                    outputColumnName: "CreatedTimeFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "ParentTitle",
                    outputColumnName: "ParentTitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(
                    inputColumnName: "ParentDescription",
                    outputColumnName: "ParentDescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate(
                    "Features", "TypeFeaturized", "TitleFeaturized",
                    "DescriptionFeaturized", "EnvironmentFeaturized", "ReporterFeaturized", "VersionFeaturized",
                    "PriorityFeaturized", "OriginalEstimateFeaturized", "CreatedTimeFeaturized",
                    "ParentTitleFeaturized", "ParentDescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);
        }
    }
}
