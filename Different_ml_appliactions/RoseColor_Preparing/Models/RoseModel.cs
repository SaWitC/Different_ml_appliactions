using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML;

namespace Rose_Clustering.Models
{
    public class RoseModel
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;  
    }
}
