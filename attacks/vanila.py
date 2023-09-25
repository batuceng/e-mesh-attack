from .attack import Attack


class VANILA(Attack):
    r"""
    Vanila version of Attack.
    It just returns the input data.

    """

    def __init__(self, model, device, seed=3):
        super().__init__('VANILA', model, device, seed)
        self.supported_mode = ['default']
        self.targeted = None

    def attack(self, data, labels):
        r"""
        Overridden.
        """
        adv_images = data.clone().detach().to(self.device)
        return adv_images
