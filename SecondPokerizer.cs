using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace poker_estimator
{
    internal class SecondPokerizer
    {
        private const long Day = 8 * 60 * 60;
        private static readonly ImmutableList<ulong> PokerKeys =
            new List<ulong> { 1, 2, 3, 5, 8, 13, 20, 40, 100 }
                .ToImmutableList();
        private static readonly ImmutableDictionary<ulong, string> PokerDictionary =
            PokerKeys.ToImmutableDictionary(
                key => key * Day,
                key => key.ToString());
    
        private readonly ulong? _seconds;

        public SecondPokerizer(string seconds)
        {
            try
            {
                _seconds = ulong.Parse(seconds);
            }
            catch (Exception)
            {
                _seconds = null;
            }
        }

        public string ToPokerDays()
        {
            var pokerKey = PokerDictionary.Keys
                .Cast<ulong?>()
                .Where(pokerValue => _seconds.HasValue && _seconds.Value <= pokerValue)
                .DefaultIfEmpty(null)
                .Min();
            return pokerKey.HasValue ? PokerDictionary[pokerKey.Value] : "?";
        }
    }
}