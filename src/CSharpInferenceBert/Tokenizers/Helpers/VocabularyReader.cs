using System.Collections.Generic;
using System.IO;

namespace BERTTokenizers.Helpers
{
    public class VocabularyReader
    {
        public static List<string> ReadFile(string filsename)
        {
            string longString = "";
            string filename =  "C:/repos/AndroidBert/AndroidBert/AndroidBert/Model/Vocabularies/Vocab.txt";
            var result = new List<string>();
            string 
            if (!File.Exists(@filename)) 
            {
            }
            using (var reader = new StreamReader(filename))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        result.Add(line);
                    }
                }
            }

            return result;
        }
    }
}
