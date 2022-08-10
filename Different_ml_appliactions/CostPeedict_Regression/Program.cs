using CostPeedict_Regression.Models;
using Microsoft.ML;
using System;
using System.IO;

namespace CostPeedict_Regression
{
    class Program
    {
        static string _trainDataPath = @"C:\Users\USER\Downloads\Different_ml_appliactions\Different_ml_appliactions\CostPeedict_Regression\Files\taxi-fare-train.csv";
        static string _testDataPath = @"C:\Users\USER\Downloads\Different_ml_appliactions\Different_ml_appliactions\CostPeedict_Regression\Files\taxi-fare-test.csv";
        static string _modelPath = @"C:\Users\USER\Downloads\Different_ml_appliactions\Different_ml_appliactions\CostPeedict_Regression\Files\Model.zip";

        private static MLContext _mlContext = new MLContext(seed: 0);
        static void Main(string[] args)
        {
            var model = Train(_mlContext, _trainDataPath);

            var trainDataShematic = GetShematic(_trainDataPath, _mlContext).Schema;


            Evalute(_mlContext,model);

            TestSinglePrediction(_mlContext, model);

            SaveModel(_mlContext,model,trainDataShematic);
        }

        private static IDataView GetShematic(string path,MLContext mLContext)
        {
            WriteMess(ConsoleColor.Green, "Load Data", ConsoleColor.White);

            var dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(path, hasHeader: true);

            WriteMess(ConsoleColor.Yellow, "Load Data Complete", ConsoleColor.White);
            return dataView;
        }
        private static ITransformer Train(MLContext mLContext ,string path)
        {
            WriteMess(ConsoleColor.Green, "Train", ConsoleColor.White);

            //string ppp = Path.Combine(Environment.GetCommandLineArgs()[0], "..", "..", "..", "Data", "issues_train.tsv");
            var dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, hasHeader: true);

            var pipelane = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName:"FareAmount")//fare amount is a number  because we do not need use MapValueToKey
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName:"PaymentTypeEincoded", inputColumnName: "PaymentType"))//eincoding 
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEincoded", inputColumnName: "RateCode"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEincoded", inputColumnName: "VendorId"))
                .Append(mLContext.Transforms.Concatenate("Features", "VendorIdEincoded", "RateCodeEincoded", "PassengerCount", "TripDistance", "RateCodeEincoded"))
                .Append(mLContext.Regression.Trainers.FastTree());

            WriteMess(ConsoleColor.Yellow, "Train Complete", ConsoleColor.White);
            return pipelane.Fit(dataView);
        }

        private static void Evalute(MLContext mLContext,ITransformer transformerModel)
        {
            WriteMess(ConsoleColor.Green, "Evalute", ConsoleColor.White);

            IDataView dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = transformerModel.Transform(dataView);
            var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
            WriteMess(ConsoleColor.Yellow, "Evalute Writed", ConsoleColor.White);
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            WriteMess(ConsoleColor.Green, "Testing with single model", ConsoleColor.White);

            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxTripPrediction>(model);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
            WriteMess(ConsoleColor.Yellow, "Testing with single model Complete", ConsoleColor.White);
        }

        private static void SaveModel(MLContext mLContext,ITransformer transformerModel,DataViewSchema TrainDataShema)
        {
            mLContext.Model.Save(transformerModel, TrainDataShema, _modelPath);
            WriteMess(ConsoleColor.Green, "Model saved", ConsoleColor.White);
        }

        private static void WriteMess(ConsoleColor messageColor,string message ,ConsoleColor nextColor)
        {
            Console.WriteLine(message, Console.ForegroundColor = messageColor); Console.ForegroundColor =nextColor;
        }
    }
}
