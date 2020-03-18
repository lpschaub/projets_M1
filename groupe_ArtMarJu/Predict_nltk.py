import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 


class Predict(object) : 


	def __init__(self, doc, seuil =0.5, pos = 0, neg =0) : 

		self.doc = doc
		self.pos = pos
		self.neg = neg
		self.seuil = seuil


	def workspace(self) : 
		
		def score(indice, liste) :
			val = 0
			no, intens = False, False
			try :
				if liste[i-1] == "so" :
					intens = True
				for y in range(i-3,i) :
					if liste[y] in ["n't", "not", "might", "no"]:
						no = True
					elif liste[y] == "could" :
						if liste[y+1]=="have" and liste[y+2]=="been" :
							no = True
					elif liste[y] in ["absolutely", "really", "very", "completely", "truly", "incredibly"] :
							intens = True
				if no == True :
					val = -1
				else :
					if intens == True :
						val = 2
					else :
						val = 1		
			except IndexError :
				val = 1 
			return val
	
		lemmatizer = WordNetLemmatizer() 
		
		new_doc = (self.doc).replace("<", " ")
		new_doc = new_doc.replace(">", " ")
		new_doc = new_doc.lower()
		
		for i in ["don't watch", "don't go", "stay away", "should have", "should not have", "don't waste", "too long", "too much"] :
			if i in new_doc :
				self.seuil-=0.1
						   
		tok = word_tokenize(new_doc)
		
		for i in range (0,len(tok)):
			if lemmatizer.lemmatize(tok[i], pos ="n") in ["performance", "talent", "wisdom", "success", "triumph", "satisfaction", "quality", "pleasure", "perfection", "originality", "maturity", "masterpiece", "majesty", "magnificence", "intelligence", "ingenuity", "humility", "humor", "humour", "hooray", "high-quality", "harmony", "greatness", "fulfillment", "gladly", "excitement", "felicity", "exultation", "excellence", "energy", "elegance","efficiently", "dynamic", "cuteness", "delight", "champion", "charisma", "brilliance", "brilliantly", "bravo", "breakthrough", "attraction", "astonishment", "benefit", "bravery", "bonus", "admiration", "agility", "accomplishment", "beautifully", "beauty",] :
				try :
					if (tok[i-1]=="of") and (tok[i-2])=="lack" :
						self.neg+=1
						continue
				except IndexError :
					pass
				self.seuil+=score(i,tok)
			if lemmatizer.lemmatize(tok[i], pos ="n") in ["mess", "stupidity", "rubbish", "shit", "downhill", "mush","sabotage", "crass", "nonsense", "tastelessness", "jerk", "crap", "mediocrity", "achingly", "gruesome", "relentlessly", "garbage", "badly", "fiasco", "painfully", "junk", "frustration", "downwards", "flaw", "headache", "error","malaise", "grossly", "unfortunately", "problem", "disappointment", "detriment","abomination", "absurdity", "emptiness", "bullshit", "desperation", "deception", "calamity", "carnage"] :
				self.seuil-=score(i,tok)
			if lemmatizer.lemmatize(tok[i], pos ="a") in ["good", "genuine", "smart", "nice", "funny", "fun", "authentic", "incredible", "cult", "sensational", "worthy", "wonderful", "useful", "beloved", "warm", "unforgettable", "remarkable", "thankful", "talented", "superb", "sumptuous", "spontaneous", "sublime", "stylish", "strong", "sincere", "splendid", "revolutionary", "respectful", "remarkable", "realistic", "rightful", "powerful", "prodigious", "priceless", "precious", "poignant", "poetic", "pleasant", "phenomenal", "mind-blowing", "meticulous", "memorable", "marvellous", "marvelous", "heartwarming", "majestic", "magical", "magnificent", "lawful", "keen", "joyful", "jubilant", "irresistible", "inventive", "intelligent", "inspirational", "innovative", "ingenious", "imaginative", "ideal", "idyllic", "honorable", "hilarious", "graceful", "gorgeous", "glorious", "genial", "formidable", "flawless", "favorite", "faithful", "fantastic", "fancy", "extraordinary", "fabulous", "exemplar", "examplary", "eye-catching", "enthusiastic", "enjoyable", "energetic", "eloquent", "elegant", "efficacious", "efficient", "delightful", "cute", "creative", "beautiful", "courageous", "cool", "constructive", "classy", "colorful", "chic", "cheerful", "charismatic", "brilliant", "bright", "breathtaking", "brave", "awesome", "attractive", "blessed", "better-than-expected", "beneficent", "believable", "ambitious", "amusing", "admirable", "agreeable", "excellent", "great", "best"] :
				self.seuil+=score(i,tok)
			if lemmatizer.lemmatize(tok[i], pos ="a") in ["bad", "rude", "ludicrous", "inaccurate", "pointless", "deplorable", "tasteless", "wooden", "joyless", "tedious", "lethargic", "implausible", "flat", "predictable", "ridiculous", "incomprehensible", "underdeveloped", "crappy", "sleazy", "amateurish", "cheesy", "painful", "sucker", "reluctant", "lackluster", "unpleasant", "useless", "unremarkable", "unlikeable", "abysmal", "unlovable", "unrealistic", "terrible", "stupid", "snob", "pathetic", "shitty", "reactionary", "racist", "outrageous", "mediocre", "dull", "horrendous", "cringy", "overdone", "irrelevant", "low", "lows", "junky", "exagerated", "hating", "messy", "insupportable", "infamous", "infernal", "inexcusable","inappropriate", "inelegant", "inefficient", "ineloquent", "erratic", "ignoble", "awful", "lame", "idiot", "horrible", "hideous", "grotesque", "excessive", "absurd", "empty", "disrespectful", "detestable", "despicable", "catastrophic", "delusional", "chaotic", "cheap", "deceptive", "wrong", "abominable", "weak", "down", "unnecessary", "apathetic", "akward", "banal"] :
				try :
					if (tok[i] == "bad") and (tok[i+1] == "guy") :
						continue;
				except IndexError :
					pass
				self.seuil-=score(i,tok)
			if lemmatizer.lemmatize(tok[i], pos ="v") in ["love", "dare", "interest", "encourage", "succeed", "worth", "accomplish", "calm", "charm", "comfort", "diversify", "flatter", "distinguish", "astound", "passionate", "satisfy", "thrive", "impress", "illuminate", "glow", "fascinate", "flourish", "stun", "excel", "gratify", "eyecatch", "exult", "excite", "exalt", "entertain", "enrich", "empower", "dominate", "dignify", "congratulate", "cherish", "cheer", "captivate", "convince", "celebrate", "appeal", "astonish", "approve", "appreciate", "applaud", "amaze", "admire","enjoy", "recommend", "adore", "frustrate"] :
				try : 
					if tok[i] == "love" and tok[i-1] == "in" :
						continue;
				except IndexError :
					pass
				self.seuil+=score(i,tok)
			if lemmatizer.lemmatize(tok[i], pos ="v") in ["hate", "dislike", "confuse", "exagerate", "waste", "endure", "depress", "detest", "ruin", "bore", "disgust", "creak", "botch", "struggle", "disappoint", "lack", "fail", "agonize", "disturb", "annoy", "exasperate", "boycott", "cringe", "deceive", "deplore", "embarrass"] :
				self.seuil-=score(i,tok)

					   

	def predict(self) :

		self.workspace()

		if self.seuil <= 0.5 : 
			self.predicted = 'neg'
		else : 
			self.predicted = 'pos'


if __name__ == '__main__':

	pred = Predict(open('../corpus/imdb/neg/10847_4.txt').read())
	pred.predict()
	print(pred.doc)
	print(pred.pos)
	print(pred.neg)
	print (pred.predicted)