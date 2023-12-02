import argparse
import aphrodite.endpoints.openai.api_server as openai_server
import aphrodite.endpoints.kobold.api_server as kobold_server


def main():
    parser = argparse.ArgumentParser(description='Aphrodite Engine CLI')
    subparsers = parser.add_subparsers()

    # Create the parser for the "openai" command
    serve_parser = subparsers.add_parser('openai',
                                         help='Start the Aphrodite API server')
    openai_server.make_parser(serve_parser)
    serve_parser.set_defaults(func=lambda args: openai_server.run_server(args))

    # Create the parser for the "kobold" command
    kobold_parser = subparsers.add_parser('kobold',
                                          help='Start the Kobold API server')
    kobold_server.make_parser(kobold_parser)
    kobold_parser.set_defaults(
        func=lambda args: kobold_server.run_server(args))

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
