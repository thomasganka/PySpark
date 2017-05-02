from pyspark.sql import HiveContext
from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.ml.feature import IDF as MLIDF
from pyspark.ml.feature import Tokenizer, RegexTokenizer
import re
from pyspark.sql.functions import regexp_replace, trim, col, lower, explode, split, udf
from pyspark.sql.types import *
from pyspark.ml.feature import StopWordsRemover

HiveContext = HiveContext(sc)
pline = HiveContext.table("analytics_pline_v1p_tbl")
pline.registerTempTable("analytics_pline_v1p_tbl")
documents = HiveContext.sql(" \
select \
site_id as account_id \
,concat(year,month,day) as dt \
,text \
from analytics_pline_v1p_tbl \
where concat(year,month,day) IN ('20160912') \
and type = 0 \
and lower(written_by) != 'info' \
and lower(text) not like ('encrypted-text%') \
and length(cast(text as string)) < 2000 \
")

#words.printSchema()

df = (documents
  .rdd
  .toDF())

cdf = df.select(col("account_id"),col("dt"),col("text"),lower(regexp_replace(col("text"), "\p{Punct}", "")).alias("clean_text"))
#cdf.show(100)
tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
wdf = tokenizer.transform(cdf)
remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=["a", "also", "ah", "am", "an", "and", "are", "as", "at",
      "be", "been", "but", "by", "bye", "can", "do", "does", "er", "eh", "for",
      "from", "had", "has", "have", "hello", "hi", "i", "if", "il", "im",
      "in", "into", "is", "it", "its", "iv", "ive", "me", "mine", "mr",
      "mrs", "ms", "my", "nbsp", "of", "on", "or", "our", "ours", "so",
      "thank", "thanks", "that", "thats", "the", "them", "then", "there",
      "they", "theyre", "theyve", "this", "thus", "to", "ty", "um", "up",
      "ur", "us", "use", "was", "we", "were", "weve", "with", "you",
      "your", "youre", "yes", "please", "thankyou", "service", "dont", 
      "get", "now", "yall", "customer", "just", "says", "take", "long",
      "okay","much", "will", "go", "make", "ok", "oh","like", "almost", "around", "back",
      "th", "per", "want", "every", "time", "everytime", "able", "many", "need", "know", "wanted",
      "sure", "sounds", "don", "t", "yeah", "won", "let", "ve", "cant", "wont", "another", "find", "see",
      "doesnt", "got", "ill", "left", "keeps", "saying", "nice", "day", "really", "appreciate",
      "still", "theres", "trying", "set", "u", "wait", "till", "whats", "wondering", "went", "guys",
      "hasnt", "no", "not", "what", "how", "would", "de", "one", "all", "help", "when", "que", "out", 
	  "good", "any", "about", "great", "new", "only", "did", "could", "should", "again", "looking", 
	  "think", "today", "why", "sorry", "well", "more", "try", "which", "where", "same", "because",
	  "para", "check", "name", "merci", "pas", "vous"])
swdf = remover.transform(wdf)
words = (swdf
    .select(col("account_id"), col("dt"), explode(col("filtered")).alias("word")))

words.registerTempTable("words")
#through Hive
HiveContext.sql("""
    SELECT word,count(*) as cnt
    FROM words
	WHERE word != ''
	AND length(word) > 2
    GROUP BY word
	ORDER BY cnt DESC""").show(10)
#55 secs
#through spark DF
(words
	.groupBy("word")
	.count()
	.sort(col("count").desc())).show(100)



#ML TF
htf = MLHashingTF(inputCol="words", outputCol="features")
tf = htf.transform(wdf)
tf.show(truncate=False)

#N-Gram
from pyspark.ml.feature import NGram
ngram = NGram(inputCol="words", outputCol="ngrams")
ngramDataFrame = ngram.transform(wordsDataFrame)
for ngrams_label in ngramDataFrame.select("ngrams", "account_id").take(3):
  print(ngrams_label)
# 
#htf = MLHashingTF(inputCol="features", outputCol="tf")
#tf = htf.transform(df)
#tf.show(truncate=False)
#
#res = tf.rdd.map(lambda x : (x.account_id,x.features,x.tf,(None if x.tf is None else x.tf.values.sum())))
#
#for r in res.take(10):
#    print r