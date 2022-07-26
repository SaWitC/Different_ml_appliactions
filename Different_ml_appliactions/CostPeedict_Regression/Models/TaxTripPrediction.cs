using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CostPeedict_Regression.Models
{
    public class TaxTripPrediction
    {
        [ColumnName("Score")]
        public float FareAmount { get; set; }
    }
}
