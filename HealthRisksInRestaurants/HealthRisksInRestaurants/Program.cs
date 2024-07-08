using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace HealthRisksInRestaurants
{
    #region model input class
    public class ModelInput
    {
        [LoadColumn(12)]
        [ColumnName(@"inspection_score")]
        public string Inspection_score { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"inspection_type")]
        public string Inspection_type { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"violation_description")]
        public string Violation_description { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"risk_category")]
        public string Risk_category { get; set; }

    }

    #endregion

    /// <summary>
    /// model output class for HealthRisksInRestaurantsPrediction.
    /// </summary>
    #region model output class
    public class ModelOutput
    {
        [ColumnName(@"inspection_score")]
        public float[] Inspection_score { get; set; }

        [ColumnName(@"inspection_type")]
        public float[] Inspection_type { get; set; }

        [ColumnName(@"violation_description")]
        public float[] Violation_description { get; set; }

        [ColumnName(@"risk_category")]
        public uint Risk_category { get; set; }

        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; }

    }

    #endregion
    class Program
    {
        static void Main(string[] args)
        {
            string MLNetModelPath = Path.GetFullPath("C:/Users/katar/OneDrive/Desktop/SIR1/ML.NET Project/HealthRisksInRestaurants/HealthRisksInRestaurantsPrediction_ConsoleApp/HealthRisksInRestaurantsPrediction.mlnet");
            PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

            var mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            var sampleData = new ModelInput();

            Console.WriteLine("Enter inspection type: ");
            sampleData.Inspection_type = Console.ReadLine();

            Console.WriteLine("Enter violation description: ");
            sampleData.Violation_description = Console.ReadLine();

            Console.WriteLine("Enter inspection score: ");
            sampleData.Inspection_score = Console.ReadLine();

            var prediction = _predictionEngine.Predict(sampleData);


            ////// Print Prediction
            Console.WriteLine("\n\n---------------Prediction------------------");
            Console.WriteLine($"Inspection type: {sampleData.Inspection_type}");
            Console.WriteLine($"Violation description: {sampleData.Violation_description}");
            Console.WriteLine($"Inspection score: {sampleData.Inspection_score}");
            Console.WriteLine($"Predicted risk category: {prediction.PredictedLabel}");
            Console.ReadKey();
        }
    }
    
}
