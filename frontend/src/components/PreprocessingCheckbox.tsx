import type { CheckboxStates } from '../interface/PreprocessingTypes';

interface PreprocessingCheckboxProps {
    sections: {
        title: string,
        options: {
            id: string,
            label: string,
        }[]
        dropdown?: {
            options: string[],
            value: string,
            onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void
        }
    }[]
    checkboxes: CheckboxStates,
    handleCheckboxChange: (event: React.ChangeEvent<HTMLInputElement>) => void
}

const PreprocessingCheckbox = ({sections, checkboxes, handleCheckboxChange}: PreprocessingCheckboxProps) => {
    return (
        <div className="space-y-8 w-full">
        {sections.map((section) => (
          <div key={section.title}>
            <h2 className="text-2xl font-bold text-secondary mb-4">{section.title}</h2>
            <div className="space-y-4">
              {section.options.map((option) => (
                <div key={option.id} className="flex items-center">
                  <input
                    type="checkbox"
                    id={option.id}
                    name={option.id}
                    checked={checkboxes[option.id as keyof CheckboxStates]}
                    onChange={handleCheckboxChange}
                    className="h-5 w-5 rounded border-gray-300 text-accent focus:ring-accent"
                  />
                  <label htmlFor={option.id} className="ml-3 text-gray-700">
                    {option.label}
                  </label>
                  {section.dropdown && (
                    <select
                      value={section.dropdown.value}
                      onChange={section.dropdown.onChange}
                      className="ml-4 p-2 border border-gray-300 rounded-md"
                    >
                      {section.dropdown.options.map((col: string) => (
                        <option key={col} value={col}>
                          {col} (dropdown)
                        </option>
                      ))}
                    </select>
                  
                  )}
                </div>
              ))}
              <hr className="my-6 border-t border-gray-300" />
              </div>
            </div>
          ))}
        </div>
    )
}

export default PreprocessingCheckbox;