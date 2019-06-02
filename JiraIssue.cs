using Microsoft.ML.Data;

namespace poker_estimator
{
    public class JiraIssue
    {
        public string Key { get; set; }
        public string Id { get; set; }
        public string Parent { get; set; }
        public string Type { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public string Time { get; set; }

    }
}