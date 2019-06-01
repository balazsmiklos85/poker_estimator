using Microsoft.ML.Data;

namespace poker_estimator
{
    public class JiraIssue
    {
        [LoadColumn(0)]
        public string Key { get; set; }
        [LoadColumn(1)]
        public string Id { get; set; }
        [LoadColumn(2)]
        public string Parent { get; set; }
        [LoadColumn(3)]
        public string Summary { get; set; }
        [LoadColumn(4)]
        public string Time { get; set; }

    }
}