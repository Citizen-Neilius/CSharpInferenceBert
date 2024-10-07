using BERTTokenizers.Base;

namespace BERTTokenizers
{
    public class BertUncasedBaseTokenizer : UncasedTokenizer
    {
        public BertUncasedBaseTokenizer() : base($@"C:\Workspace\Models\bert-large-uncased-whole-word-masking-finetuned-squad\vocab.txt")
        {
        }
    }
}
