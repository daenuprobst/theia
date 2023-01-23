class BulmaTabs {
    constructor(tabContainerId) {
        this.tabContainerId = tabContainerId;
        this.tabContainer = document.getElementById(tabContainerId);
        this.init();
    }

    init() {
        this.tabs = [...this.tabContainer.children];
        this.tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.setActive(tab);
            });
        });
    }

    setActive(tab) {
        [...this.tabContainer.children].forEach(tab => {
            tab.classList.remove('is-active');

            let targetId = tab.getAttribute('data-target-id');
            let target = document.getElementById(targetId);
            if (target) {
                target.classList.remove('is-active');
            }
        });

        tab.classList.add('is-active');

        let targetId = tab.getAttribute('data-target-id');
        let target = document.getElementById(targetId);
        if (target) {
            target.classList.add('is-active');
        }
    }

    setActiveByIndex(index) {
        this.setActive(this.tabs[index]);
    }
}