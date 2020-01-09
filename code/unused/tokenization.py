import six
import collections
import unicodedata
import os

VOCAB_NAME = 'vocab.txt'

def convert_to_unicode(text):
	"""Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, int):
			return str(text)
		elif isinstance(text, float):
			return str(text)
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text.decode("utf-8", "ignore")
		elif isinstance(text, unicode):
			return text
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
	"""Returns text encoded in a way suitable for print or `tf.logging`."""

	# These functions want `str` for both Python2 and Python3, but in one case
	# it's a Unicode string and in the other it's a byte string.
	if six.PY3:
		if isinstance(text, str):
			return text
		elif isinstance(text, bytes):
			return text.decode("utf-8", "ignore")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	elif six.PY2:
		if isinstance(text, str):
			return text
		elif isinstance(text, unicode):
			return text.encode("utf-8")
		else:
			raise ValueError("Unsupported string type: %s" % (type(text)))
	else:
		raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
	vocab = collections.OrderedDict()
	index = 0
	with open(vocab_file, 'r', encoding='utf-8') as reader:
		while True:
			token = convert_to_unicode(reader.readline())
			if not token:
				break
			token = token.strip()
			vocab[token] = index
			index += 1
	return vocab

def whitespace_tokenize(text):
	text = text.strip()
	if not text:
		return []
	tokens = text.split()
	return tokens

class FullTokenizer(object):
	def __init__(self, vocab_file, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[AUG]")):
		if not os.path.isfile(vocab_file):
			raise ValueError("can't find vocab file at {}".format(vocab_file))
		self.vocab = load_vocab(vocab_file)
		self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
		self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
		self.ids_to_tokens = collections.OrderedDict(
			[(ids, tok) for tok, ids in self.vocab.items()])
	def tokenize(self, text):
		split_tokens = []
		for token in self.basic_tokenizer.tokenize(text):
			for sub_token in self.wordpiece_tokenizer.tokenize(token):
				split_tokens.append(sub_token)
		return split_tokens
	def convert_tokens_to_ids(self, tokens):
		ids = []
		for token in tokens:
			ids.append(self.vocab[token])
		return ids
	def convert_ids_to_tokens(self, ids):
		tokens = []
		for i in ids:
			tokens.append(self.ids_to_tokens[i])
		return tokens
	def save_vocabulary(self, vocab_path):
		"""Save the tokenizer vocabulary to a directory or file."""
		index = 0
		if os.path.isdir(vocab_path):
			vocab_file = os.path.join(vocab_path, VOCAB_NAME)
		with open(vocab_file, "w", encoding="utf-8") as writer:
			for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
				if index != token_index:
					logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
								   " Please check that the vocabulary is not corrupted!".format(vocab_file))
					index = token_index
				writer.write(token + u'\n')
				index += 1
		return vocab_file

class BasicTokenizer(object):
	def __init__(self, do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
		self.do_lower_case = do_lower_case
		self.never_split = never_split
	def tokenize(self, text):
		text = convert_to_unicode(text)
		text = self._clean_text(text)
		orig_tokens = whitespace_tokenize(text)
		split_tokens = []
		for token in orig_tokens:
			if self.do_lower_case and token not in self.never_split:
				token = token.lower()
				token = self._run_strip_accents(token)
			split_tokens.extend(self._run_split_on_punc(token))
		output_tokens = whitespace_tokenize(" ".join(split_tokens))
		return output_tokens
	def _run_strip_accents(self, text):
		text = unicodedata.normalize('NFD',text)
		output = []
		for char in text:
			cat = unicodedata.category(char)
			if cat == 'Mn':
				continue
			output.append(char)
		return "".join(output)
	def _run_split_on_punc(self, text):
		if text in self.never_split:
			return [text]
		chars = list(text)
		i = 0
		start_new_word = True
		output = []
		while i < len(chars):
			char = chars[i]
			if _is_punctuation(char):
				output.append([char])
				start_new_word = True
			else:
				if start_new_word:
					output.append([])
				start_new_word = False
				output[-1].append(char)
			i += 1
		return ["".join(x) for x in output]
	def _clean_text(self, text):
		output = []
		for char in text:
			cp = ord(char)
			if cp == 0 or cp == 0xfffd or _is_control(char):
				continue
			if _is_whitespace(char):
				output.append(" ")
			else:
				output.append(char)
		return "".join(output)

class WordpieceTokenizer(object):
	def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
		self.vocab = vocab
		self.unk_token = unk_token
		self.max_input_chars_per_word = max_input_chars_per_word
	def tokenize(self, text):
		text = convert_to_unicode(text)
		output_tokens = []
		for token in whitespace_tokenize(text):
			chars = list(token)
			if len(chars) > self.max_input_chars_per_word:
				output_tokens.append(self.unk_token)
				continue
			is_bad = False
			start = 0
			sub_tokens = []
			while start < len(chars):
				end = len(chars)
				cur_substr = None
				while start < end:
					substr = "".join(chars[start:end])
					if start > 0:
						substr = "##" + substr
					if substr in self.vocab:
						cur_substr = substr
						break
					end -= 1
				if cur_substr is None:
					is_bad = True
					break
				sub_tokens.append(cur_substr)
				start = end
			if is_bad:
				output_tokens.append(self.unk_token)
			else:
				output_tokens.extend(sub_tokens)
		return output_tokens

def _is_whitespace(char):
	"""Checks whether `chars` is a whitespace character."""
	# \t, \n, and \r are technically contorl characters but we treat them
	# as whitespace since they are generally considered as such.
	if char == " " or char == "\t" or char == "\n" or char == "\r":
		return True
	cat = unicodedata.category(char)
	if cat == "Zs":
		return True
	return False


def _is_control(char):
	"""Checks whether `chars` is a control character."""
	# These are technically control characters but we count them as whitespace
	# characters.
	if char == "\t" or char == "\n" or char == "\r":
		return False
	cat = unicodedata.category(char)
	if cat.startswith("C"):
		return True
	return False


def _is_punctuation(char):
	"""Checks whether `chars` is a punctuation character."""
	cp = ord(char)
	# We treat all non-letter/number ASCII as punctuation.
	# Characters such as "^", "$", and "`" are not in the Unicode
	# Punctuation class but we treat them as punctuation anyways, for
	# consistency.
	if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
			(cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
		return True
	cat = unicodedata.category(char)
	if cat.startswith("P"):
		return True
	return False










































































