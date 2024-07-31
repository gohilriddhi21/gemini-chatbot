import yaml

class ConfigExtraction:
    def __init__(self):
        try:
            with open("config/config.yml") as file:
                self.config = yaml.safe_load(file)

            self.AI_STUDIO_KEY = self.config['keys']['ai_studio_key']
        except yaml.YAMLError as err:
            print(err)

if __name__ == '__main__':
    print(ConfigExtraction().AI_STUDIO_KEY)
