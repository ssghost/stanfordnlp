"""
basic demo script
"""

import argparse
import os

import stanfordnlp
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/stanfordnlp_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--lang', help='Demo language',
                        default="en")
    parser.add_argument('-c', '--cpu', action='store_true', help='Use cpu as the device.')
    args = parser.parse_args()

    example_sentences = {"en": "Donald Trump will make America great again. He promised to build a wall for the security of all citizens.",
            "zh": "從前有座山，山上有座廟，廟裡有個老和尚，帶著一個小和尚。有一天，小和尚下山去化緣，老和尚對他說，路邊的野花不要採。",
            "fr": "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie. Il tente d'abord de faire carrière comme marchand d'art chez Goupil & C.",
            "vi": "Trận Trân Châu Cảng (hay Chiến dịch Hawaii theo cách gọi của Bộ Tổng tư lệnh Đế quốc Nhật Bản) là một đòn tấn công quân sự bất ngờ được Hải quân Nhật Bản thực hiện nhằm vào căn cứ hải quân của Hoa Kỳ tại Trân Châu Cảng thuộc tiểu bang Hawaii vào sáng Chủ Nhật, ngày 7 tháng 12 năm 1941, dẫn đến việc Hoa Kỳ sau đó quyết định tham gia vào hoạt động quân sự trong Chiến tranh thế giới thứ hai."}

    if args.lang not in example_sentences:
        print(f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
        exit()

    # download the models
    stanfordnlp.download(args.lang, args.models_dir, confirm_if_exists=True)
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    pipeline = stanfordnlp.Pipeline(models_dir=args.models_dir, lang=args.lang, use_gpu=(not args.cpu))
    # process the document
    doc = pipeline(example_sentences[args.lang])
    # access nlp annotations
    print('')
    print('Input: {}'.format(example_sentences[args.lang]))
    print("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))
    print('---')
    print('tokens of first sentence: ')
    doc.sentences[0].print_tokens()
    print('')
    print('---')
    print('dependency parse of first sentence: ')
    doc.sentences[0].print_dependencies()
    print('')

