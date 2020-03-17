# KRISTINA SOPHIE CAMILLE

import re
#import spacy
#nlp=spacy.load('en_core_web_sm')
class Predict(object) :


	def __init__(self, doc, nlp, seuil = 0.5) :

		self.doc = doc
		self.doc_min=self.doc.lower()
		self.seuil = seuil
		self.nlpdoc = nlp(self.doc)
		self.lemmas = [token.lemma_ for token in self.nlpdoc]
		self.lemmatized=" ".join(self.lemmas)

	def workspace(self) :

		mots_pos=["perfect","smart","sincere","pleasant","suspense","accurate","startling","charisma","go see","rich","modern","charismatic","intense","strong","love","light","lit","suggest","original","relevant","intriguing","beautiful","likable","engaging","impressive","sensual","genuine","fresh","mysterious","authentic","interesting","fun","nice","favourite","favorite","funny","colourful","inventive","innovative","moving","entertaining","intelligent","amusing","nicely","good","solid","memorable","impressed","enjoyable","enjoy","notable"]
		mots_pos_plus=["astounding","astound","astonishing","astonish","convincingly","poignant","heartfelt","adventurous","captivating","captivate","captivated","exciting","excitment","excited","imaginative","talented","talent","spicy","effective","audacious","bold","well-written","upbeat","stunning","well-acted","impactful","impact","applause","crisp","suspenseful","deep","mind-bending","incredible","explosive","unbelievable","mesmerizing","immersive","mesmerising","heartwarming","underrated","scary af","hypnotic","riveting","groundbreaking","cool","touch","touching","exquisite","splendid","fabulous","superb","should see","beautifully","should watch","recommend","fascinating","fascinate","fascinated","unforgettable","hilarious","fantastic","thrilling","recommended","amazing","excellent","excellently","delightful","marvellous","brilliant","brilliantly","transcendant","delight","finest","joyous","dazzling","genius","masterfully","great","greatest","masterpiece","awesome","sublime","wonderfully","wonderful","excellently","witty","irresistible","must-see","must see","praise","outstanding","breathtaking","terrific","convincing","convinced","substance"]
		expression_pos=["well written","great performance","worth watching","worth seeing","work of art","one of the best","some of the best","charming"]
		mots_neg=["soppy","cheesy","predict","drag","mindless","weakness","bored","bore","dissatisfied","mistake","problem","embarassing","embarass","endless","mess","odd","typical","disturbing","half \w+ time","half of \w+ time","wrong","merely","inept","annoying","grotesque","sink","bother","apparently","brooding","unlikable","depressing","off putting","ugly","weird","unfortunately","ridiculous","worse","stupid","predictable","silliest","disappoint","laughable","screw","sketchy","non-existent","more or less","in \w+ opinion","too much","too many","too little","too few","attempt","long","skip","absence","poor","idiot"]
		mots_neg_plus=["sucky","suck","bullshit","cringy","corny","messy","ludicrous","idiotic","dumb","incompetence","incompetent","stink","forgettable","crap","message-less","unpleasant","irrelevant","retarded","implausible","sadly","unfunny","achingly","horrible","atrocityu","sappy","anemic","slow-moving","lack","flat","frivilous","non-relevant","hollow","nonsense","unsophisticated","non-sense","embarassment","dreary","plot hole","plot-holes","plot-hole","miscast","useless","gratuitous","awful","shameless","worn-out","wear out","cliched","deadly","hodgepodge","hotchpotch","ludicrous","uninteresting","lifeless","limp","absurd","nonsensical","non-sensical","go nowhere","rubbish","garbage","disaster","disatrous","rambling","unacceptable","dreadful","horrid","crappy","terrible","badly","bad","poorly","appal","insult","insulting","appalling","vomit","futility","gross","shit","pathetic","disappointing","disappointed","hate","dislike","failure","mediocre","mediocrity","worst","fail","cliche","cliché","avoid","painful","painfully","cornball","downer","irritate","vile","whiny","overrated","faux pas","boring","vacuous","pretentious","lame","miserably","zero effort","brainless","issue","make no sense","inaccurate","cringy","shallow","repetitive","tedious","empty","monotone","monotonous"]
		expression_neg=["does not work","doesn't work","clich ?","clich?","no clue","overly","come on","not enough","should not watch","shouldn't watch","dull","do not watch","don't watch","overdone","screeching","at best","that is about it","ladies and gentlemen","for some reason","could have been","should have been","here and there","hard to watch","nothing more than","over and over","the problem is that","waste","the whole thing","to begin with","might as well","what the hell","the whole movie","too bad that","saving grace","could not stand","couldn't stand","can not stand","no idea","cannot stand","can't stand","not to say","the worst","below average""sunken ship"]
		intensifiers=re.compile(r'really|truly|very|incredibly| so |highly|extremely|just|utterly|totally|the most')
		minimisers=re.compile(r'slightly|somewhat|somehow|kind| bit ')
		for mot in mots_pos:
			match=re.findall(rf"[^ ]+ [^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^{mot} ", self.lemmatized)
			for m in match:
				if ("not " in m or "no " in m or "neither " in m or "only" in m or "would be" in m or "could have been" in m or "would have" in m or "than" in m or "might be" in m or "lack" in m) and "not only" not in m:
					self.seuil-=0.2
					#print(m+"-1")
				else:
					if re.search(intensifiers, m):
						self.seuil+=0.3
					elif re.search(minimisers, m):
						self.seuil+=0.07
					else:
						self.seuil+=0.15
					#print(m+"+1")
		for mot in mots_pos_plus:
			match=re.findall(rf"[^ ]+ [^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^{mot} ", self.lemmatized)
			for m in match:
				if ("not " in m or "no " in m or "neither " in m or "only" in m or "would be" in m or "could have been" in m or "would have" in m or "than" in m or "might be" in m or "lack" in m) and "not only" not in m:
					self.seuil-=0.2
					#print(m+"-2")
				else:
					if re.search(intensifiers, m):
						self.seuil+=0.4
					elif re.search(minimisers, m):
						self.seuil+=0.15
					else:
						self.seuil+=0.3
					#print(m+"+2")
		for mot in expression_pos:
			match=re.findall(rf"[^ ]+ [^ ]+ [^ ]+ {mot} ", self.doc_min)+re.findall(rf"^[^ ]+ [^ ]+ {mot} ", self.doc_min)+re.findall(rf"^[^ ]+ {mot} ", self.doc_min)+re.findall(rf"^{mot} ", self.doc_min)
			for m in match:
				if ("not " in m or "no " in m or "neither " in m or "only" in m or "would be" in m or "could have been" in m or "would have" in m or "than" in m or "far from" in m) and "not only" not in m:
					self.seuil-=0.3
					#print(m+"-2")
				else:
					if re.search(intensifiers, m):
						self.seuil+=0.5
					else:
						self.seuil+=0.3
					#print(m+"+2")


		for mot in mots_neg:
			match=re.findall(rf"[^ ]+ [^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^{mot} ", self.lemmatized)
			for m in match:
				if ("not " in m or "no " in m or "neither " in m ) and "not only" not in m:
					self.seuil+=0.05
					#print(m+"-1")
				else:
					if re.search(intensifiers, m):
						self.seuil-=0.4
					elif re.search(minimisers, m):
						self.seuil-=0.1
					else:
						self.seuil-=0.2
					#print(m+"+1")
		for mot in mots_neg_plus:
			match=re.findall(rf"[^ ]+ [^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ [^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^[^ ]+ {mot} ", self.lemmatized)+re.findall(rf"^{mot} ", self.lemmatized)
			for m in match:
				if ("not " in m or "no " in m or "neither " in m ) and "not only" not in m:
					self.seuil+=0.05
					#print(m+"-2")
				else:
					if re.search(intensifiers, m):
						self.seuil-=0.6
					elif re.search(minimisers, m):
						self.seuil-=0.15
					else:
						self.seuil-=0.3
					#print(m+"+2")
		for mot in expression_neg:
			match=re.findall(rf"{mot}", self.doc_min)
			for m in match:
				self.seuil-=0.4
					#print(m+"+2")
		bonne_note=re.compile(r'[6-9]((/10)|( out of 10))')
		mauvaise_note=re.compile(r'[0-5]((/10)|( out of 10))')
		if re.search(bonne_note, self.doc_min):
			self.seuil+=0.2
			#print("note explicite positive dans : ")
			#print(self.doc_min)
		if re.search(mauvaise_note, self.doc_min):
			self.seuil-=0.4
			#print("note explicite negative dans : ")
			#print(self.doc_min)


	def predict(self) :

		self.workspace()
		#print(self.seuil)

		if self.seuil < 0.5 :
			self.predicted = 'neg'
		else :
			self.predicted = 'pos'


if __name__ == '__main__':


	pred = Predict(open('../corpus/imdb/pos/33_7.txt').read())
	pred.predict()
	print (pred.predicted)
