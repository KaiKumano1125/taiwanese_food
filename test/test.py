class Dog:
    def __init__(self, name, age):
        self.name=name
        self.age=age
    def bark(self):
        print(f"{self.name} wow")

my_dog=Dog("pochi", 3)
my_dog.bark()
