using System;

namespace poker_estimator
{
    public class ThreePointEstimation
    {
        public decimal Min { get; set; }
        public decimal Max { get; set; }
        public decimal Mean { get; set; }
        public decimal? Time { get; set; }

        public void Estimate()
        {
            var expectedSeconds = (Min + 4 * Mean + Max) * SecondPokerizer.Day / 6;
            var pokerDays = new SecondPokerizer(((ulong)expectedSeconds).ToString()).ToPokerDays();
            Time = decimal.Parse(pokerDays);
        }
    }
}