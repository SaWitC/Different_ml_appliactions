using Microsoft.ML;
using Rose_Clustering.Models;
using System;
using System.Collections.Generic;
using System.IO;

namespace Rose_Clustering
{
    class Program
    {
        private static string _dataPath = "C:\\Users\\USER\\Downloads\\Different_ml_appliactions\\Different_ml_appliactions\\RoseColor_Preparing\\Files\\iris.data";
        private static string _modelPath = "C:\\Users\\USER\\Downloads\\Different_ml_appliactions\\Different_ml_appliactions\\RoseColor_Preparing\\Files\\IrisClusteringModel.zip";
        //class action
        static void Main(string[] args)
        {
            //create mlContext
            var mlContext = new MLContext(0);
            // load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<RoseModel>(_dataPath,hasHeader:false,separatorChar:',');

            string featuresColumnName = "Features";



            var pipeline = mlContext.Transforms.Concatenate(featuresColumnName, "SepalLength", "SepalWidth",
                "PetalLength", "PetalWidth").Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName,null,3));

            //training
            var model = pipeline.Fit(dataView);

            //saving model

            using (FileStream fs =new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fs);
            }
            //created predictor
            var predictor = mlContext.Model.CreatePredictionEngine<RoseModel,RosePrediction>(model,dataView.Schema);

            //prediction
            var prediction = predictor.Predict(Setosa);
            // write data
            Console.WriteLine(prediction.PredictedClusterId);
            Console.WriteLine(String.Join(" ",prediction.Distances));
           




        }

        private static readonly RoseModel Setosa = new RoseModel
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };
    }
}
