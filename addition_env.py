from data_generator import generate
from utils import train, get_accuracy


class AdditionEnvironment:
    def __init__(self, batch_size=4096, seq_len=9, number_of_digits=(1, 9), seed=1):
        """
        number_of_digits specifies the difficulty of the task within the curriculum (it's # of non-zero digits)
        each curriculum should have a fixed seq_len. Higher seq_len's define harder curriculums (larger networks mainly)
        """
        assert ((type(number_of_digits) == tuple and number_of_digits[0] <= number_of_digits[1] <= seq_len) or
                (type(number_of_digits == int) and 0 < number_of_digits <= seq_len)), "Wrong environment parameters"

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.number_of_digits = number_of_digits
        self.seed = seed

    def train_epoch(self, model, encoder_optimizer, decoder_optimizer, criterion, epoch_length=10,
                    eval_everything=None, validate_using=None):
        """
        Trains the model using epoch_length batches of size specified by the generator kwargs
        Returns the average loss during the epoch, the average training accuracy, and a validation accuracy at the end
        The accuracy contains both a per digit accuracy, and a per operation accuracy.
        eval_everything, if set to a triplet of type (min, max, xxx), specifies that xxx test examples should be used to
        evaluate the accuracy of the model on sequences of length varying from min to max. if set to xxx: min, max=1, 9
        validate_using is the number of examples of the same number of digits to validate on (if None use batch size).
        if eval_everything is not None, validate_using is not used
        """
        assert model.decoder.seq_len == 1 + self.seq_len, "{} and {} don't match".format(model.decoder.seq_len,
                                                                                         self.seq_len)

        if validate_using is None:
            validate_using = self.batch_size
        losses = 0
        per_digit_acs = 0
        per_number_acs = 0
        for step in range(epoch_length):
            inputs, labels = generate(self.batch_size, self.number_of_digits, self.seq_len, seed_n=self.seed)
            loss, (per_digit_ac, per_number_ac) = train(model, encoder_optimizer,
                                                        decoder_optimizer, criterion, inputs, labels)

            losses += loss
            per_digit_acs += per_digit_ac
            per_number_acs += per_number_ac

        if validate_using > 0 and eval_everything is None:
            test_inputs, test_labels = generate(validate_using, self.number_of_digits, self.seq_len)
            test_accuracy = get_accuracy(model(test_inputs), test_labels)
        else:
            test_accuracy = (0., 0.)

        test_results = {}
        if eval_everything is not None:
            if isinstance(eval_everything, tuple):
                min_n, max_n, n_eval_examples = eval_everything
            else:
                min_n = 1
                max_n = 9
                n_eval_examples = eval_everything
            test_inputs, test_labels = zip(*(generate(n_eval_examples, n_dig, self.seq_len)
                                             for n_dig in range(min_n, max_n + 1)))
            test_results = {n_dig: get_accuracy(model(test_inputs[n_dig - min_n]), test_labels[n_dig - min_n])
                            for n_dig in range(min_n, max_n + 1)}

        return (losses / epoch_length, (per_digit_acs / epoch_length, per_number_acs / epoch_length),
                test_accuracy, test_results)
