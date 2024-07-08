﻿
// This file was auto-generated by ML.NET Model Builder. 

using System;

namespace HealthRisksInRestaurantsPrediction_ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            HealthRisksInRestaurantsPrediction.ModelInput sampleData = new HealthRisksInRestaurantsPrediction.ModelInput()
            {
                Inspection_score = @"92",
                Inspection_type = @"Routine - Unscheduled",
                Violation_description = @"Moderate risk food holding temperature",
            };


            Console.WriteLine("Using model to make single prediction -- Comparing actual Risk_category with predicted Risk_category from sample data...\n\n");


            Console.WriteLine($"Inspection_score: {@"92"}");
            Console.WriteLine($"Inspection_type: {@"Routine - Unscheduled"}");
            Console.WriteLine($"Violation_description: {@"Moderate risk food holding temperature"}");
            Console.WriteLine($"Risk_category: {@"Moderate Risk"}");


            var sortedScoresWithLabel = HealthRisksInRestaurantsPrediction.PredictAllLabels(sampleData);
            Console.WriteLine($"{"Class",-40}{"Score",-20}");
            Console.WriteLine($"{"-----",-40}{"-----",-20}");

            foreach (var score in sortedScoresWithLabel)
            {
                Console.WriteLine($"{score.Key,-40}{score.Value,-20}");
            }
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
