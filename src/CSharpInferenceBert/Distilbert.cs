using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Diagnostics;
namespace AndroidBert
{
    public struct BertInput
    {
        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public long[] TypeIds { get; set; }
    }
    internal class Distilbert
    {
        public void Go()
        {
            try
            {
                var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";
                Console.WriteLine(sentence);

                // Create Tokenizer and tokenize the sentence.
                var tokenizer = new BertBaseTokenizer();

                // Get the sentence tokens.
                var tokens = tokenizer.Tokenize(sentence);
                // Console.WriteLine(String.Join(", ", tokens));

                // Encode the sentence and pass in the count of the tokens in the sentence.
                var encoded = tokenizer.Encode(tokens.Count(), sentence);

                // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
                var bertInput = new BertInput()
                {
                    InputIds = encoded.Select(t => t.InputIds).ToArray(),
                    AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                    TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                };
                // Get path to model to create inference session.
                var modelPath = $@"{AppDomain.CurrentDomain.BaseDirectory}\Model\model.onnx";
                //var modelPath = @"C:\code\bert-nlp-csharp\BertNlpTest\BertNlpTest\bert-large-uncased-finetuned-qa.onnx";

                using var runOptions = new RunOptions();
                using var session = new InferenceSession(modelPath);

                // Create input tensors over the input data.
                using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                      new long[] { 1, bertInput.InputIds.Length });

                using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                      new long[] { 1, bertInput.AttentionMask.Length });

                using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                      new long[] { 1, bertInput.TypeIds.Length });

                // Create input data for session. Request all outputs in this case.
                var inputs = new Dictionary<string, OrtValue>
              {
                  { "input_ids", inputIdsOrtValue },
                  { "input_mask", attMaskOrtValue },
                  { "segment_ids", typeIdsOrtValue }
              };
                using var output = session.Run(runOptions, inputs, session.OutputNames);
                // Get the Index of the Max value from the output lists.
                // We intentionally do not copy to an array or to a list to employ algorithms.
                // Hopefully, more algos will be available in the future for spans.
                // so we can directly read from native memory and do not duplicate data that
                // can be large for some models
                // Local function
                int GetMaxValueIndex(ReadOnlySpan<float> span)
                {
                    float maxVal = span[0];
                    int maxIndex = 0;
                    for (int i = 1; i < span.Length; ++i)
                    {
                        var v = span[i];
                        if (v > maxVal)
                        {
                            maxVal = v;
                            maxIndex = i;
                        }
                    }
                    return maxIndex;
                }

                var startLogits = output[0].GetTensorDataAsSpan<float>();
                int startIndex = GetMaxValueIndex(startLogits);

                var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
                int endIndex = GetMaxValueIndex(endLogits);

                var predictedTokens = tokens
                              .Skip(startIndex)
                              .Take(endIndex + 1 - startIndex)
                              .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                              .ToList();

                // Print the result.
                Console.WriteLine(String.Join(" ", predictedTokens));

            }
            catch (Exception ex )
            {
                Debug.WriteLine(ex);
            }
        }
    }
}
