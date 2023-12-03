using NeoCortexApi.Entities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeoCortexApi.Classifiers
{
    public class ClassificationAndDistance : IComparable<ClassificationAndDistance>
    {
        public string Classification { get; private set; }
        public int Distance { get; private set; }
        public int ClassificationNo { get; private set; }

        public ClassificationAndDistance(string classification, int distance, int classificationNo)
        {
            Classification = classification;
            Distance = distance;
            ClassificationNo = classificationNo;
        }

        public int CompareTo(ClassificationAndDistance other)
        {
            if (other == null)
            {
                return 1; // If 'other' is null, this instance is considered greater.
            }

            if (Distance < other.Distance)
            {
                return -1;
            }
            else if (Distance > other.Distance)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
    }

    public class KNeighborsClassifier<TIN, TOUT> : IClassifier<TIN, TOUT>
    {
        private Dictionary<string, List<int[]>> models = new Dictionary<string, List<int[]>>();
        private int numberOfNeighbors = 4;
        private int sdrs = 20;

        private int CalculateMinimumDistance(int[] classifiedSequence, int unclassifiedIndex)
        {
            int shortestDistance = unclassifiedIndex;

            foreach (var classifiedIdx in classifiedSequence)
            {
                var distance = Math.Abs(classifiedIdx - unclassifiedIndex);
                shortestDistance = Math.Min(shortestDistance, distance);
            }

            return shortestDistance;
        }

        private Dictionary<int, int> GenerateDistanceTable(int[] classifiedSequence, int[] unclassifiedSequence)
        {
            var distanceTable = new Dictionary<int, int>();

            foreach (var index in unclassifiedSequence)
            {
                distanceTable[index] = CalculateMinimumDistance(classifiedSequence, index);
            }

            return distanceTable;
        }

        public List<ClassifierResult<TIN>> GetPredictedInputValues(Cell[] unclassifiedCells, short howMany = 1)
        {
            if (unclassifiedCells.Length == 0)
            {
                return new List<ClassifierResult<TIN>>();
            }

            var unclassifiedSequence = unclassifiedCells.Select(idx => idx.Index).ToArray();
            var mappedElements = new Dictionary<int, List<ClassificationAndDistance>>();

            foreach (var model in models)
            {
                foreach (var (sequence, idx) in model.Value.Select((seq, idx) => (seq, idx)))
                {
                    var distanceTable = GenerateDistanceTable(sequence, unclassifiedSequence);
                    foreach (var dict in distanceTable)
                    {
                        if (!mappedElements.ContainsKey(dict.Key))
                        {
                            mappedElements[dict.Key] = new List<ClassificationAndDistance>();
                        }
                        mappedElements[dict.Key].Add(new ClassificationAndDistance(model.Key, dict.Value, idx));
                    }
                }
            }

            foreach (var mappings in mappedElements)
            {
                mappings.Value.Sort();
            }

            // Considering 'numberOfNeighbors' for KNN by taking the first 'numberOfNeighbors' elements
            var result = mappedElements.SelectMany(x => x.Value)
                .Take(numberOfNeighbors)
                .Select(cad => new ClassifierResult<TIN>
                {
                    PredictedInput = (TIN)Convert.ChangeType(cad.Classification, typeof(TIN)),
                    // Assuming similarity and NumOfSameBits should be 0 for now
                    Similarity = 0,
                    NumOfSameBits = 0
                })
                .Take(howMany)
                .ToList();

            return result;
        }


        public void Learn(TIN input, Cell[] cells)
        {
            var classification = input as string;
            int[] cellIndices = cells.Select(idx => idx.Index).ToArray();

            if (!models.ContainsKey(classification))
            {
                models[classification] = new List<int[]>();
            }

            if (!models[classification].Exists(seq => Enumerable.SequenceEqual(seq, cellIndices)))
            {
                if (models[classification].Count > sdrs)
                {
                    models[classification].RemoveAt(0);
                }
                models[classification].Add(cellIndices);
            }
        }

        public void ClearState()
        {
            models.Clear();
        }
    }
}
