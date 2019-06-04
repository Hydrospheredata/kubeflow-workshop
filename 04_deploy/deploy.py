import argparse
import os, urllib.parse
from hydrosdk import sdk


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-version', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--application-name-postfix', required=True)
    parser.add_argument('--hydrosphere-address', required=True)
    parser.add_argument('--dev', help='Flag for development purposes', action="store_true")
    
    args = parser.parse_args()

    application_name = f"{args.model_name}{args.application_name_postfix}"
    model = sdk.Model.from_existing(args.model_name, args.model_version)
    application = sdk.Application.singular(application_name, model)
    result = application.apply(args.hydrosphere_address)
    print(result, flush=True)

    with open("./application_name.txt" if args.dev else "/application_name.txt", "w+") as file:
        file.write(application_name)

    with open("./application_link.txt" if args.dev else "/application_link.txt", "w+") as file:
        file.write(urllib.parse.urljoin(args.hydrosphere_address, f"applications/{application_name}"))