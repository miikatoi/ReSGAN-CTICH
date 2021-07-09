from models.GAN_model import GANModel


def create_model(opt, data):
    return GANModel(opt, data)


def get_cmd_parser_modifier(parser):
    cmd_parser_modifier = GANModel.modify_cmd_parser
    return cmd_parser_modifier


def get_args(parser):
    cmd_parser_modifier = GANModel.modify_cmd_parser
    cmd_parser_modifier(parser)
