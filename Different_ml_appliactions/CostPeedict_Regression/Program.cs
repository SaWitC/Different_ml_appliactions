using CostPeedict_Regression.Models;
using Microsoft.ML;
using System;
using System.IO;

namespace CostPeedict_Regression
{
    class Program
    {

        static string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Files", "taxi-fare-train.csv");
        static string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Files", "taxi-fare-test.csv");
        static string _modelPath = Path.Combine(Environment.CurrentDirectory, "Files", "Model.zip");

        private static MLContext _mlContext = new MLContext(seed: 0);
        static void Main(string[] args)
        {
            var model = Train(_mlContext, _trainDataPath);

            Evalute(_mlContext,model);
        }

        private static ITransformer Train(MLContext mLContext ,string path)
        {
            var dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, hasHeader: true);

            var pipelane = mLContext.Transforms.CopyColumns("Label", "FareAmount")//fare amount is a number  because we do not need use MapValueToKey
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("PaymentTypeEincoded", "PaymentType"))//eincoding 
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("RateCodeEincoded", "RateCode"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding("VendorIdEincoded", "VendorId"))
                .Append(mLContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mLContext.Regression.Trainers.FastTree());

            return pipelane.Fit(dataView);
        }

        private static void Evalute(MLContext mLContext,ITransformer transformerModel)
        {
            var TestDataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true);

            var predictions = transformerModel.Transform(TestDataView);

            var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine("=================================================");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine("=================================================");
            Console.WriteLine("=================================================");
        }

        
    }
}
