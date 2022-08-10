using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Rose_Clustering.Models
{
    class RosePrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
