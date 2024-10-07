using BERTTokenizers.Base;

namespace BERTTokenizers
{
    public class BertBaseTokenizer : CasedTokenizer
    {
        public BertBaseTokenizer() : base("C:/repos/AndroidBert/AndroidBert/AndroidBert/Vocabularies/base_cased.txt")
        {
        }
    }
}
