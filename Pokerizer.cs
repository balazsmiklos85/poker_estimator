using System;

internal class Pokerizer
{
    private double? time;

    public Pokerizer(string time)
    {
        try
        {
            this.time = ulong.Parse(time) / 28800.0d;
        }
        catch (Exception)
        {
            this.time = null;
        }
    }

    public string ToPokerValue()
    {
        if (time == null)
            return "?";
        if (time == 0.0d)
            return "0";
        if (time <= 1.0d)
            return "1";
        if (time <= 2.0d)
            return "2";
        if (time <= 3.0d)
            return "3";
        if (time <= 5.0d)
            return "5";
        if (time <= 8.0d)
            return "8";
        if (time <= 13.0d)
            return "13";
        if (time <= 20.0d)
            return "20";
        if (time <= 40.0d)
            return "40";
        if (time <= 100.0d)
            return "100";
        return "?";
    }
}