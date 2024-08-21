import React from 'react';

interface StarRatingProps {
    rating: number;
}

const starMap: { [key: number]: JSX.Element } = {
    0: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
        </svg>
    ),
    1: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
            <path d="M10.6904 16.02C11.1289 16.02 11.5674 16.02 12.0059 16.02C12.7513 16.02 11.7993 16.078 11.5514 16.1276C11.1719 16.2035 10.6409 16.2666 10.3436 16.5342C9.97989 16.8615 10.1312 17.0034 10.6067 16.8571C11.2411 16.6619 12.0172 16.7734 12.6756 16.7734C13.5359 16.7734 13.9624 17.3001 14.7146 17.6344C15.2402 17.868 15.5045 18.5885 15.9643 18.8184" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
        </svg>
    ),
    2: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
            <path d="M10.6904 16.02C11.1289 16.02 11.5674 16.02 12.0059 16.02C12.7513 16.02 11.7993 16.078 11.5514 16.1276C11.1719 16.2035 10.6409 16.2666 10.3436 16.5342C9.97989 16.8615 10.1312 17.0034 10.6067 16.8571C11.2411 16.6619 12.0172 16.7734 12.6756 16.7734C13.5359 16.7734 13.9624 17.3001 14.7146 17.6344C15.2402 17.868 15.5045 18.5885 15.9643 18.8184" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2871 17.6344C16.2871 17.1063 15.9624 16.5553 15.8565 16.0259C15.7236 15.3613 14.9881 15.1284 14.3915 14.9975C13.6477 14.8342 12.4739 15.0984 11.8263 14.7284C11.6446 14.6245 11.8754 14.0166 12.0356 13.9809C12.6285 13.8492 13.4218 13.975 14.0268 13.975C14.6687 13.975 15.2704 13.7597 15.8565 13.7597" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
        </svg>
    ),
    3: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
            <path d="M11.3362 14.2978C10.119 14.0949 9.2856 12.7672 8.16707 12.3605C7.57784 12.1462 6.58811 11.8849 7.43757 12.7671C7.81739 13.1615 8.43017 14.0074 8.43017 14.5669C8.43017 15.0141 8.76146 15.8749 8.40625 16.3189C8.2384 16.5287 8.0137 17.0304 7.99964 17.3115C7.98738 17.5568 8.06937 17.8963 7.94583 18.1187C7.84798 18.2948 7.80512 18.3888 7.67675 18.5492C7.44232 18.8423 7.53092 18.6195 7.70067 18.4954C8.26431 18.0835 8.85187 17.5984 9.34503 17.2038C9.53568 17.0513 9.80894 16.9206 9.94298 16.7195C10.1717 16.3764 11.1557 16.3689 10.6904 16.0199" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M10.6904 16.02C11.1289 16.02 11.5674 16.02 12.0059 16.02C12.7513 16.02 11.7993 16.078 11.5514 16.1276C11.1719 16.2035 10.6409 16.2666 10.3436 16.5342C9.97989 16.8615 10.1312 17.0034 10.6067 16.8571C11.2411 16.6619 12.0172 16.7734 12.6756 16.7734C13.5359 16.7734 13.9624 17.3001 14.7146 17.6344C15.2402 17.868 15.5045 18.5885 15.9643 18.8184" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2871 17.6344C16.2871 17.1063 15.9624 16.5553 15.8565 16.0259C15.7236 15.3613 14.9881 15.1284 14.3915 14.9975C13.6477 14.8342 12.4739 15.0984 11.8263 14.7284C11.6446 14.6245 11.8754 14.0166 12.0356 13.9809C12.6285 13.8492 13.4218 13.975 14.0268 13.975C14.6687 13.975 15.2704 13.7597 15.8565 13.7597" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
        </svg>
    ),
    4: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
            <path d="M11.3362 14.2978C10.119 14.0949 9.2856 12.7672 8.16707 12.3605C7.57784 12.1462 6.58811 11.8849 7.43757 12.7671C7.81739 13.1615 8.43017 14.0074 8.43017 14.5669C8.43017 15.0141 8.76146 15.8749 8.40625 16.3189C8.2384 16.5287 8.0137 17.0304 7.99964 17.3115C7.98738 17.5568 8.06937 17.8963 7.94583 18.1187C7.84798 18.2948 7.80512 18.3888 7.67675 18.5492C7.44232 18.8423 7.53092 18.6195 7.70067 18.4954C8.26431 18.0835 8.85187 17.5984 9.34503 17.2038C9.53568 17.0513 9.80894 16.9206 9.94298 16.7195C10.1717 16.3764 11.1557 16.3689 10.6904 16.0199" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M10.6904 16.02C11.1289 16.02 11.5674 16.02 12.0059 16.02C12.7513 16.02 11.7993 16.078 11.5514 16.1276C11.1719 16.2035 10.6409 16.2666 10.3436 16.5342C9.97989 16.8615 10.1312 17.0034 10.6067 16.8571C11.2411 16.6619 12.0172 16.7734 12.6756 16.7734C13.5359 16.7734 13.9624 17.3001 14.7146 17.6344C15.2402 17.868 15.5045 18.5885 15.9643 18.8184" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2871 17.6344C16.2871 17.1063 15.9624 16.5553 15.8565 16.0259C15.7236 15.3613 14.9881 15.1284 14.3915 14.9975C13.6477 14.8342 12.4739 15.0984 11.8263 14.7284C11.6446 14.6245 11.8754 14.0166 12.0356 13.9809C12.6285 13.8492 13.4218 13.975 14.0268 13.975C14.6687 13.975 15.2704 13.7597 15.8565 13.7597" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M8.96777 13.1139C10.1497 13.1139 11.3317 13.1139 12.5136 13.1139C13.4113 13.1139 14.2548 12.6833 15.1087 12.6833C15.4834 12.6833 15.8581 12.6833 16.2329 12.6833C16.8693 12.6833 17.0884 12.4118 17.662 12.1691C17.9061 12.0658 18.0293 11.7639 18.2779 11.7087C18.7221 11.61 18.7622 11.3445 18.7622 10.9612" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M18.3322 11.0689C17.8489 11.0689 17.5146 11.2898 17.0825 11.3858C16.5786 11.4978 16.1017 11.4994 15.5817 11.4994C13.9154 11.4994 12.2491 11.4994 10.5828 11.4994C9.40625 11.4994 7.95863 11.6205 6.86354 11.1227C6.24739 10.8427 5.33931 10.9613 4.66309 10.9613" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M7.78418 13.1139C7.26459 13.0562 6.77024 12.5248 6.27734 12.3605" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M17.6861 10.5307C17.4861 10.5307 17.3057 10.4981 17.1479 10.6384C17.1109 10.6713 16.8946 10.9244 17.0164 10.8297C17.3378 10.5797 16.787 10.6319 16.6337 10.7699C16.1608 11.1955 15.5325 11.4655 14.8877 11.4994C14.2892 11.5309 13.6813 11.4994 13.0819 11.4994C12.1756 11.4994 11.1061 11.519 10.2356 11.2841C9.51485 11.0896 8.83083 10.8777 8.13085 10.6623C7.88279 10.5859 7.56995 10.5674 7.35352 10.4231" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2873 9.88495C15.9635 9.92093 15.5647 10.2054 15.2588 10.3155C14.894 10.4468 14.7658 10.5897 14.3559 10.6922C13.6281 10.8741 12.8757 11.052 12.1136 11.0689C11.1811 11.0896 9.46251 10.7395 8.64551 11.2842" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M9.61412 11.3918C9.45552 10.8631 8.15738 10.6509 7.67676 10.5308" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
        </svg>

    ),
    5: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M11.7499 4.5C11.6589 4.5 11.4349 4.525 11.3159 4.763L9.48992 8.414C9.20092 8.991 8.64392 9.392 7.99992 9.484L3.91192 10.073C3.64192 10.112 3.54992 10.312 3.52192 10.396C3.49692 10.477 3.45692 10.683 3.64292 10.861L6.59892 13.701C7.06992 14.154 7.28392 14.807 7.17192 15.446L6.47592 19.456C6.43292 19.707 6.58992 19.853 6.65992 19.903C6.73392 19.959 6.93192 20.07 7.17692 19.942L10.8319 18.047C11.4079 17.75 12.0939 17.75 12.6679 18.047L16.3219 19.941C16.5679 20.068 16.7659 19.957 16.8409 19.903C16.9109 19.853 17.0679 19.707 17.0249 19.456L16.3269 15.446C16.2149 14.807 16.4289 14.154 16.8999 13.701L19.8559 10.861C20.0429 10.683 20.0029 10.476 19.9769 10.396C19.9499 10.312 19.8579 10.112 19.5879 10.073L15.4999 9.484C14.8569 9.392 14.2999 8.991 14.0109 8.413L12.1829 4.763C12.0649 4.525 11.8409 4.5 11.7499 4.5ZM6.94692 21.5C6.53392 21.5 6.12392 21.37 5.77292 21.114C5.16692 20.67 4.86992 19.937 4.99892 19.199L5.69492 15.189C5.72092 15.04 5.66992 14.889 5.55992 14.783L2.60392 11.943C2.05992 11.422 1.86492 10.652 2.09492 9.937C2.32692 9.214 2.94092 8.697 3.69792 8.589L7.78592 8C7.94392 7.978 8.07992 7.881 8.14792 7.743L9.97492 4.091C10.3119 3.418 10.9919 3 11.7499 3C12.5079 3 13.1879 3.418 13.5249 4.091L15.3529 7.742C15.4219 7.881 15.5569 7.978 15.7139 8L19.8019 8.589C20.5589 8.697 21.1729 9.214 21.4049 9.937C21.6349 10.652 21.4389 11.422 20.8949 11.943L17.9389 14.783C17.8289 14.889 17.7789 15.04 17.8049 15.188L18.5019 19.199C18.6299 19.938 18.3329 20.671 17.7259 21.114C17.1109 21.565 16.3099 21.626 15.6309 21.272L11.9779 19.379C11.8349 19.305 11.6639 19.305 11.5209 19.379L7.86792 21.273C7.57592 21.425 7.26092 21.5 6.94692 21.5Z" fill="#FFE45A"/>
            <path d="M11.3362 14.2978C10.119 14.0949 9.2856 12.7672 8.16707 12.3605C7.57784 12.1462 6.58811 11.8849 7.43757 12.7671C7.81739 13.1615 8.43017 14.0074 8.43017 14.5669C8.43017 15.0141 8.76146 15.8749 8.40625 16.3189C8.2384 16.5287 8.0137 17.0304 7.99964 17.3115C7.98738 17.5568 8.06937 17.8963 7.94583 18.1187C7.84798 18.2948 7.80512 18.3888 7.67675 18.5492C7.44232 18.8423 7.53092 18.6195 7.70067 18.4954C8.26431 18.0835 8.85187 17.5984 9.34503 17.2038C9.53568 17.0513 9.80894 16.9206 9.94298 16.7195C10.1717 16.3764 11.1557 16.3689 10.6904 16.0199" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M10.6904 16.02C11.1289 16.02 11.5674 16.02 12.0059 16.02C12.7513 16.02 11.7993 16.078 11.5514 16.1276C11.1719 16.2035 10.6409 16.2666 10.3436 16.5342C9.97989 16.8615 10.1312 17.0034 10.6067 16.8571C11.2411 16.6619 12.0172 16.7734 12.6756 16.7734C13.5359 16.7734 13.9624 17.3001 14.7146 17.6344C15.2402 17.868 15.5045 18.5885 15.9643 18.8184" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2871 17.6344C16.2871 17.1063 15.9624 16.5553 15.8565 16.0259C15.7236 15.3613 14.9881 15.1284 14.3915 14.9975C13.6477 14.8342 12.4739 15.0984 11.8263 14.7284C11.6446 14.6245 11.8754 14.0166 12.0356 13.9809C12.6285 13.8492 13.4218 13.975 14.0268 13.975C14.6687 13.975 15.2704 13.7597 15.8565 13.7597" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M8.96777 13.1139C10.1497 13.1139 11.3317 13.1139 12.5136 13.1139C13.4113 13.1139 14.2548 12.6833 15.1087 12.6833C15.4834 12.6833 15.8581 12.6833 16.2329 12.6833C16.8693 12.6833 17.0884 12.4118 17.662 12.1691C17.9061 12.0658 18.0293 11.7639 18.2779 11.7087C18.7221 11.61 18.7622 11.3445 18.7622 10.9612" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M18.3322 11.0689C17.8489 11.0689 17.5146 11.2898 17.0825 11.3858C16.5786 11.4978 16.1017 11.4994 15.5817 11.4994C13.9154 11.4994 12.2491 11.4994 10.5828 11.4994C9.40625 11.4994 7.95863 11.6205 6.86354 11.1227C6.24739 10.8427 5.33931 10.9613 4.66309 10.9613" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M7.78418 13.1139C7.26459 13.0562 6.77024 12.5248 6.27734 12.3605" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M17.6861 10.5307C17.4861 10.5307 17.3057 10.4981 17.1479 10.6384C17.1109 10.6713 16.8946 10.9244 17.0164 10.8297C17.3378 10.5797 16.787 10.6319 16.6337 10.7699C16.1608 11.1955 15.5325 11.4655 14.8877 11.4994C14.2892 11.5309 13.6813 11.4994 13.0819 11.4994C12.1756 11.4994 11.1061 11.519 10.2356 11.2841C9.51485 11.0896 8.83083 10.8777 8.13085 10.6623C7.88279 10.5859 7.56995 10.5674 7.35352 10.4231" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M16.2873 9.88495C15.9635 9.92093 15.5647 10.2054 15.2588 10.3155C14.894 10.4468 14.7658 10.5897 14.3559 10.6922C13.6281 10.8741 12.8757 11.052 12.1136 11.0689C11.1811 11.0896 9.46251 10.7395 8.64551 11.2842" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
            <path d="M9.61412 11.3918C9.45552 10.8631 8.15738 10.6509 7.67676 10.5308" stroke="#FFE45A" stroke-width="3" stroke-linecap="round"/>
        </svg>
    ),
};

const StarRating: React.FC<StarRatingProps> = ({ rating }) => {
    const roundedRating = Math.floor(rating); 

    return (
        <div>
            {starMap[roundedRating] || starMap[0]}
        </div>
    );
};

export default StarRating;

